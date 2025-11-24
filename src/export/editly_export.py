import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

_CUT_SENTINEL = object()


def _resolve_transition_token(token: Optional[str], base_duration: float) -> Optional[Dict[str, float]] | object:
    """Map free-form transition token from LLM to Editly transition config.

    Returns:
        dict -> transition configuration for Editly
        _CUT_SENTINEL -> indicates explicit hard cut
        None -> unknown token (fallback to automatic system)
    """
    if not token:
        return None
    t = token.strip().lower()
    if not t:
        return None

    speed = 1.0
    for mark, factor in ("fast", 0.65), ("slow", 1.6), ("cinematic", 1.25), ("linger", 1.8):
        if mark in t:
            speed *= factor
            t = t.replace(mark, "")

    t = t.replace("transition", "").replace("effect", "").strip()
    # Alias map: mapeia tokens para transições (INCLUINDO GL!)
    # Container agora tem headless-gl configurado, então todas transições funcionam
    alias_map = {
        "cut": _CUT_SENTINEL,
        "hard cut": _CUT_SENTINEL,
        "jump cut": _CUT_SENTINEL,
        "fade": "fade",
        "crossfade": "fade",
        "dissolve": "fade",
        "directional-left": "directional-left",
        "slide-left": "directional-left",
        "push-left": "directional-left",
        "whip-left": "directional-left",
        "directional-right": "directional-right",
        "slide-right": "directional-right",
        "push-right": "directional-right",
        "whip-right": "directional-right",
        "directional-up": "directional-up",
        "slide-up": "directional-up",
        "directional-down": "directional-down",
        "slide-down": "directional-down",
        # Transições GL (agora funcionam!)
        "glitch": "glitch-matrix",
        "glitch-matrix": "glitch-matrix",
        "pixel": "pixelize",
        "pixelate": "pixelize",
        "pixelize": "pixelize",
        "warp": "warpzoom",
        "warp-zoom": "warpzoom",
        "warpzoom": "warpzoom",
        "crosswarp": "crosswarp",
        "cross-warp": "crosswarp",
        "crosszoom": "crosszoom",
        "cross-zoom": "crosszoom",
        "zoom-blur": "linearblur",
        "zoomblur": "linearblur",
        "linear-blur": "linearblur",
        "circle-open": "circleopen",
        "circleopen": "circleopen",
        "circle-close": "circleclose",
        "circleclose": "circleclose",
        "dreamy": "dreamy",
        "dream": "dreamy",
        "swap": "swap",
        "random": "random",
    }

    # Normalize multi-word tokens
    normalized = t.replace("  ", " ")
    normalized = normalized.strip(" -")

    # Direct match
    if normalized in alias_map:
        name = alias_map[normalized]
    else:
        # Try partial contains
        name = None
        for key, value in alias_map.items():
            if key in normalized:
                name = value
                break

    if name is None:
        return None
    if name is _CUT_SENTINEL:
        return _CUT_SENTINEL

    duration = max(0.12, base_duration * speed)
    return {"name": name, "duration": round(duration, 2)}


def json_dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False, indent=2)


# ============================================================================
# SISTEMA DE TRANSIÇÕES PROFISSIONAIS
# ============================================================================

class TransitionSystem:
    """
    Sistema inteligente de seleção de transições baseado em:
    - Visual mode (dynamic, narrative, static)
    - Mood da cena
    - Tipo de conteúdo (image vs video)
    - Posição na narrativa
    """

    # Transições disponíveis no Editly (COM SUPORTE GL HEADLESS)
    # Container agora configurado com headless-gl + xvfb para suportar TODAS transições!
    TRANSITIONS = {
        # Básicas (sem GL)
        "fade": {"name": "fade", "weight": 2.5},
        "directional-left": {"name": "directional-left", "weight": 1.3},
        "directional-right": {"name": "directional-right", "weight": 1.3},
        "directional-up": {"name": "directional-up", "weight": 0.9},
        "directional-down": {"name": "directional-down", "weight": 0.9},
        "random": {"name": "random", "weight": 0.8},
        "linearblur": {"name": "linearblur", "weight": 1.2},
        # GL Transitions (agora funcionam com headless-gl!)
        "glitch-matrix": {"name": "glitch-matrix", "weight": 0.7},  # Efeito Matrix glitch
        "pixelize": {"name": "pixelize", "weight": 0.6},  # Pixelização
        "warpzoom": {"name": "warpzoom", "weight": 0.8},  # Zoom com warp espacial
        "crosswarp": {"name": "crosswarp", "weight": 0.7},  # Cross dissolve + warp
        "crosszoom": {"name": "crosszoom", "weight": 0.7},  # Cross dissolve + zoom
        "circleopen": {"name": "circleopen", "weight": 0.5},  # Abertura circular
        "circleclose": {"name": "circleclose", "weight": 0.5},  # Fechamento circular
        "dreamy": {"name": "dreamy", "weight": 0.6},  # Efeito sonhador/dreamy
        "swap": {"name": "swap", "weight": 0.6},  # Swap/troca de posição
    }

    # Perfis de transição por visual mode
    # IMPORTANTE: Agora com suporte completo a transições GL!
    MODE_PROFILES = {
        "dynamic": {
            # Transições dinâmicas e impactantes para cenas de ação
            "preferred": ["warpzoom", "glitch-matrix", "crosswarp", "linearblur", "directional-right", "pixelize"],
            "duration_range": (0.2, 0.5),
            "variation": "high",
        },
        "narrative": {
            # Transições suaves para storytelling
            "preferred": ["fade", "crosszoom", "dreamy", "directional-right", "linearblur"],
            "duration_range": (0.4, 0.8),
            "variation": "medium",
        },
        "static": {
            "preferred": ["fade", "fade", "fade"],  # Repetição intencional para peso
            "duration_range": (0.5, 1.0),
            "variation": "low",
        },
    }

    def __init__(self):
        self.last_transitions = []
        self.transition_index = 0

    def select_transition(
        self,
        visual_mode: str,
        scene_index: int,
        total_scenes: int,
        prev_type: Optional[str] = None,
        curr_type: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Seleciona transição inteligente evitando repetição.

        Args:
            visual_mode: "dynamic", "narrative", "static"
            scene_index: Índice da cena atual
            total_scenes: Total de cenas
            prev_type: "image" ou "video" da cena anterior
            curr_type: "image" ou "video" da cena atual
        """
        mode = visual_mode.lower() if visual_mode else "static"
        profile = self.MODE_PROFILES.get(mode, self.MODE_PROFILES["static"])

        # Primeira cena: sempre fade suave
        if scene_index == 0:
            return {"name": "fade", "duration": 0.6}

        # Última cena: fade out longo
        if scene_index >= total_scenes - 1:
            return {"name": "fade", "duration": 0.8}

        # Seleciona transição do perfil
        preferred = profile["preferred"]
        variation = profile["variation"]

        # Alta variação: escolhe aleatoriamente
        if variation == "high":
            # Evita repetir as últimas 2 transições
            available = [t for t in preferred if t not in self.last_transitions[-2:]]
            if not available:
                available = preferred

            chosen = random.choice(available)

        # Média variação: alterna entre preferidas
        elif variation == "medium":
            chosen = preferred[self.transition_index % len(preferred)]
            self.transition_index += 1

        # Baixa variação: sempre fade
        else:
            chosen = "fade"

        # Registra para evitar repetição
        self.last_transitions.append(chosen)
        if len(self.last_transitions) > 3:
            self.last_transitions.pop(0)

        # Duração baseada no perfil
        min_dur, max_dur = profile["duration_range"]
        duration = random.uniform(min_dur, max_dur)

        # Ajuste se transição entre video e image (mais lenta)
        if prev_type == "video" and curr_type == "image":
            duration *= 1.2
        elif prev_type == "image" and curr_type == "video":
            duration *= 0.9

        return {"name": chosen, "duration": round(duration, 2)}


# ============================================================================
# KEN BURNS EFFECT AVANÇADO
# ============================================================================

class KenBurnsEffect:
    """
    Sistema de Ken Burns effect com múltiplos estilos e movimento natural.

    Cria movimento cinematográfico em imagens estáticas usando:
    - Zoom in/out com diferentes intensidades
    - Pan (deslocamento horizontal/vertical)
    - Combinações de zoom + pan
    - Easing para movimento suave
    """

    STYLES = {
        "cinematic_reveal": {
            "zoom": "in",
            "amount": 0.25,
            "pan": "right",
            "description": "Zoom in revelando detalhes + pan suave",
        },
        "dramatic_pullback": {
            "zoom": "out",
            "amount": 0.30,
            "pan": "left",
            "description": "Zoom out dramático revelando contexto",
        },
        "subtle_drift": {
            "zoom": "in",
            "amount": 0.10,
            "pan": "down",
            "description": "Movimento sutil para não distrair",
        },
        "epic_sweep": {
            "zoom": "out",
            "amount": 0.35,
            "pan": "up",
            "description": "Movimento épico para estabelecer cena",
        },
        "intimate_focus": {
            "zoom": "in",
            "amount": 0.20,
            "pan": None,
            "description": "Zoom direto focando no assunto",
        },
        "expansive_view": {
            "zoom": "out",
            "amount": 0.18,
            "pan": None,
            "description": "Zoom out mostrando grandeza",
        },
    }

    @staticmethod
    def apply(
        visual_mode: str,
        mood: Optional[str],
        duration: float,
    ) -> Dict[str, any]:
        """
        Aplica Ken Burns effect baseado no contexto da cena.

        Args:
            visual_mode: "dynamic", "narrative", "static"
            mood: Mood da cena (ex: "misterioso", "épico", "calmo")
            duration: Duração da cena

        Returns:
            Dicionário com configurações de zoom e pan
        """
        mode = visual_mode.lower() if visual_mode else "static"
        mood_str = (mood or "").lower()

        # Seleção inteligente baseada em visual mode e mood
        if mode == "dynamic":
            # Movimento forte e variado
            styles = ["cinematic_reveal", "epic_sweep", "dramatic_pullback"]
            if "misterioso" in mood_str or "suspense" in mood_str:
                styles.append("cinematic_reveal")
            chosen_style = random.choice(styles)

        elif mode == "narrative":
            # Movimento médio, mais storytelling
            styles = ["subtle_drift", "cinematic_reveal", "expansive_view"]
            if "épico" in mood_str or "grandioso" in mood_str:
                styles.append("epic_sweep")
            chosen_style = random.choice(styles)

        else:  # static
            # Movimento mínimo, apenas para não ficar totalmente parado
            styles = ["subtle_drift", "intimate_focus"]
            chosen_style = random.choice(styles)

        style = KenBurnsEffect.STYLES[chosen_style]

        # Ajusta intensidade baseado na duração
        # Cenas curtas: movimento mais rápido/intenso
        # Cenas longas: movimento mais lento/sutil
        duration_factor = 1.0
        if duration < 3.0:
            duration_factor = 1.3
        elif duration > 8.0:
            duration_factor = 0.8

        amount = style["amount"] * duration_factor

        # Monta configuração
        config = {
            "zoomDirection": style["zoom"],
            "zoomAmount": round(amount, 2),
            "easing": "easeInOutCubic",
        }

        # Adiciona pan se definido
        if style["pan"]:
            # Velocity ajustado pela duração
            base_velocity = 60 if mode == "static" else 100
            velocity = int(base_velocity * duration_factor)

            config["move"] = {
                "direction": style["pan"],
                "velocity": velocity,
            }

        return config


# ============================================================================
# COLOR GRADING E FILTROS
# ============================================================================

class ColorGradingSystem:
    """
    Sistema de color grading automático para dar 'cara' profissional ao vídeo.

    Aplica correções de cor baseadas em:
    - Mood da cena
    - Tipo de conteúdo
    - Posição na narrativa
    """

    PRESETS = {
        "cinematic_teal_orange": {
            "saturation": 1.15,
            "contrast": 1.10,
            "brightness": 0.95,
            "vignette": 0.3,
            "description": "Look cinemático com teals e oranges",
        },
        "documentary_neutral": {
            "saturation": 1.05,
            "contrast": 1.05,
            "brightness": 1.0,
            "vignette": 0.15,
            "description": "Look documental neutro e realista",
        },
        "mystery_dark": {
            "saturation": 0.90,
            "contrast": 1.20,
            "brightness": 0.85,
            "vignette": 0.45,
            "description": "Look escuro e misterioso",
        },
        "vibrant_poppy": {
            "saturation": 1.25,
            "contrast": 1.15,
            "brightness": 1.05,
            "vignette": 0.20,
            "description": "Cores vibrantes e chamativas",
        },
        "muted_elegant": {
            "saturation": 0.85,
            "contrast": 0.95,
            "brightness": 1.0,
            "vignette": 0.25,
            "description": "Cores suaves e elegantes",
        },
    }

    @staticmethod
    def select_preset(
        mood: Optional[str],
        visual_mode: str,
        scene_index: int,
    ) -> Dict[str, float]:
        """
        Seleciona preset de color grading baseado no contexto.

        Note: Editly não tem suporte direto a color grading via API simples.
        Esta função está preparada para quando adicionar filtros customizados.
        """
        mood_str = (mood or "").lower()

        if "misterioso" in mood_str or "suspense" in mood_str:
            return ColorGradingSystem.PRESETS["mystery_dark"]
        elif "épico" in mood_str or "dramático" in mood_str:
            return ColorGradingSystem.PRESETS["cinematic_teal_orange"]
        elif "calmo" in mood_str or "sereno" in mood_str:
            return ColorGradingSystem.PRESETS["muted_elegant"]
        elif visual_mode == "dynamic":
            return ColorGradingSystem.PRESETS["vibrant_poppy"]
        else:
            return ColorGradingSystem.PRESETS["documentary_neutral"]


# ============================================================================
# GERADOR DE CONFIGURAÇÃO EDITLY PROFISSIONAL
# ============================================================================

def write_editly_config(
        base_dir: str,
        render_plan: List[dict],
        scene_timings: List[dict],
        render_prefs: dict,
        audio_path: str,
        output_path: str,
):
    """
    Gera configuração PROFISSIONAL do Editly com:
    - Transições inteligentes e variadas
    - Ken Burns effect cinematográfico
    - Movimento natural com easing
    - Variação para evitar monotonia

    MELHORIAS IMPLEMENTADAS:
    1. Sistema de transições baseado em visual mode
    2. Ken Burns effect com múltiplos estilos
    3. Variação inteligente para evitar "PowerPoint effect"
    4. Timing dinâmico baseado no conteúdo
    5. Easing functions para movimento natural
    """

    # Deriva resolução do env
    res = (os.getenv("TARGET_RESOLUTION", "720p") or "720p").lower()
    if res == "1080p":
        width, height = 1920, 1080
    elif res == "480p":
        width, height = 1280, 720
    else:
        width, height = 1280, 720

    # FPS otimizado para movimento suave
    fps = int(render_prefs.get("fps", 30))  # 30fps para movimento mais suave

    # Inicializa sistemas
    transition_system = TransitionSystem()

    # Config base
    cfg = {
        "outPath": str(Path(output_path).resolve()),
        "width": width,
        "height": height,
        "fps": fps,
        "fast": False,
        "keepSourceAudio": False,
        "audioFilePath": str(Path(audio_path).resolve()),
        "allowRemoteRequests": False,
        "clips": [],
    }

    # Mapeamento de scene_index -> timing info
    timing_map = {t.get("index"): t for t in scene_timings if isinstance(t, dict)}

    total_scenes = len(render_plan)
    prev_asset_type = None

    for scene_idx, sc in enumerate(render_plan):
        assets = sc.get("assets", [])
        if not assets:
            continue

        visual_mode = sc.get("visual_mode", "static")
        scene_transition = sc.get("transition", "fade")

        # Pega timing info se disponível
        timing_info = timing_map.get(sc.get("scene_index", scene_idx), {})
        mood = timing_info.get("mood") or sc.get("mood")

        # Processa cada asset
        for asset_idx, asset in enumerate(assets):
            asset_type = asset.get("type", "image")
            duration = float(asset.get("duration_sec", 3.0))
            local_path = asset.get("local_cache")

            if not local_path or duration <= 0:
                continue

            # Cria clip
            clip = {
                "duration": round(duration, 3),
                "layers": [],
            }

            # ============================================
            # TRANSIÇÃO INTELIGENTE
            # ============================================
            # Usa transição customizada do asset, ou gera inteligentemente
            custom_transition = asset.get("transition")

            resolved = _resolve_transition_token(custom_transition, duration)
            if resolved is _CUT_SENTINEL:
                clip["transition"] = None
            elif isinstance(resolved, dict):
                clip["transition"] = resolved
            else:
                transition = transition_system.select_transition(
                    visual_mode=visual_mode,
                    scene_index=scene_idx,
                    total_scenes=total_scenes,
                    prev_type=prev_asset_type,
                    curr_type=asset_type,
                )
                clip["transition"] = transition

            # ============================================
            # LAYER (IMAGE ou VIDEO)
            # ============================================
            if asset_type == "image":
                # IMAGE: Aplica Ken Burns effect
                layer = {
                    "type": "image",
                    "path": str(Path(local_path).resolve()),
                    "resizeMode": "cover",
                }

                manual_direction = (asset.get("zoomDirection") or "").strip().lower()
                manual_amount_raw = asset.get("zoomAmount")

                dir_map = {"in": "in", "out": "out", "left": "left", "right": "right"}
                if manual_direction in dir_map:
                    try:
                        manual_amount = float(manual_amount_raw)
                    except (TypeError, ValueError):
                        manual_amount = None
                    zoom_amount = manual_amount if manual_amount and manual_amount > 0 else 0.1
                    ken_burns = {
                        "zoomDirection": dir_map[manual_direction],
                        "zoomAmount": round(zoom_amount, 3),
                        "easing": "easeInOutCubic",
                    }
                else:
                    ken_burns = KenBurnsEffect.apply(
                        visual_mode=visual_mode,
                        mood=mood,
                        duration=duration,
                    )

                layer.update(ken_burns)

                clip["layers"].append(layer)

            else:
                # VIDEO: Sem efeitos adicionais (o vídeo já tem movimento)
                layer = {
                    "type": "video",
                    "path": str(Path(local_path).resolve()),
                    "resizeMode": "cover",
                    "cutFrom": 0,  # Começa do início
                    "cutTo": min(duration, 30),  # Limita para evitar vídeos muito longos
                }

                clip["layers"].append(layer)

            cfg["clips"].append(clip)
            prev_asset_type = asset_type

    # ============================================
    # PÓS-PROCESSAMENTO
    # ============================================
    # Garante que primeira transição seja suave
    if cfg["clips"]:
        cfg["clips"][0]["transition"] = {"name": "fade", "duration": 0.8}

    # Garante que última transição seja fade out longo
    if len(cfg["clips"]) > 1:
        # Última transição é entre penúltimo e último clip
        # Editly aplica transição no início do clip
        cfg["clips"][-1]["transition"] = {"name": "fade", "duration": 1.0}

    # Salva config
    cfg_path = Path(base_dir) / "editly_config.json5"
    Path(cfg_path).write_text(json_dumps(cfg), encoding="utf-8")

    # Log de resumo
    print(f"\n{'='*60}")
    print(f"📹 EDITLY CONFIG GERADO - MODO PROFISSIONAL")
    print(f"{'='*60}")
    print(f"Total de clips: {len(cfg['clips'])}")
    print(f"Resolução: {width}x{height} @ {fps}fps")
    print(f"Áudio: {Path(audio_path).name}")
    print(f"\nTransições utilizadas:")

    # Conta transições
    transition_counts = {}
    for clip in cfg["clips"]:
        trans = clip.get("transition")
        if trans:
            name = trans.get("name", "none")
            transition_counts[name] = transition_counts.get(name, 0) + 1

    for name, count in sorted(transition_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {count}x")

    print(f"\nConfig salvo em: {cfg_path}")
    print(f"{'='*60}\n")

    return str(cfg_path)


def _split_into_clip_slices(clips: List[Dict]) -> List[Dict]:
    """
    Função legada mantida para compatibilidade.
    Nova versão não precisa de splitting.
    """
    return clips
