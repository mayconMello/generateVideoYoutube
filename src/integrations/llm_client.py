from __future__ import annotations

import json
import os
import logging
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union

import instructor
from instructor import Instructor, Provider
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from src.pipeline.video_profiles import VideoProfile
from src.schemas.recipe import (
    Asset,
    MusicPlan,
    NarrativePlan,
    Scene,
    SceneBlueprint,
    SegmentGroupingPlan,
    VideoRecipe,
    VideoRecipeMetadata,
    VideoRecipeSceneChunk,
)

T = TypeVar("T", bound=BaseModel)

MessageParam = Union[
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
]
MessageList = Sequence[MessageParam]

logger = logging.getLogger(__name__)

DEFAULT_CHANNEL_BRAND = "Isso tem explicação"
CHANNEL_BRAND = (os.getenv("CHANNEL_BRAND_NAME") or DEFAULT_CHANNEL_BRAND).strip() or DEFAULT_CHANNEL_BRAND
BRAND_CONTEXT = (
    f"O canal '{CHANNEL_BRAND}' é especializado em dark facts, curiosidades científicas e mistérios inexplicados. "
    "Tom: curiosidade intensa + leve desconforto psicológico + fascínio científico. "
    "Público: brasileiros de 25 a 65+ anos que adoram fatos perturbadores e conhecimento que desafia o senso comum."
)


def _resolve_provider(provider: str | Provider) -> Provider:
    if isinstance(provider, Provider):
        return provider
    provider_str = (provider or "").strip().lower()
    try:
        return Provider(provider_str)
    except ValueError as exc:
        raise ValueError(f"Unsupported LLM provider '{provider_str}'.") from exc


def create_instructor_client(
    *,
    provider_override: Optional[str | Provider] = None,
    model_override: Optional[str] = None,
) -> Instructor:
    timeout = float(os.getenv("LLM_TIMEOUT", "180"))

    provider_value = provider_override or os.getenv("LLM_PROVIDER") or Provider.OPENAI.value
    provider_enum = _resolve_provider(provider_value)

    if model_override is not None:
        llm = model_override.strip()
    else:
        llm = (os.getenv("LLM_MODEL") or "gpt-5").strip()

    mode = instructor.Mode.JSON if provider_enum is Provider.OPENAI else instructor.Mode.ANTHROPIC_JSON

    return instructor.from_provider(
        f"{provider_enum.value}/{llm}",
        mode=mode,
        timeout=timeout,
    )


class TypedLLMClient:
    def __init__(self):
        self._client = create_instructor_client()

    def _build_kwargs(
        self,
        *,
        response_model: Type[T],
        messages: MessageList,
        enable_reasoning: bool = False,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            messages=list(messages),
            response_model=response_model,
            max_retries=3,
        )

        provider = self._client.provider
        if provider is Provider.OPENAI:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["reasoning_effort"] = "high" if enable_reasoning else "medium"
        if provider is Provider.ANTHROPIC:
            kwargs["max_tokens"] = 30_000
            if enable_reasoning:
                try:
                    model_name = str(self._client.kwargs.get("model", ""))
                    if "sonnet" in model_name.lower() or "opus" in model_name.lower():
                        kwargs["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": 10000
                        }
                        # Anthropic requires temperature to be exactly 1 when thinking mode is enabled.
                        kwargs["temperature"] = 1.0
                    else:
                        kwargs["temperature"] = 0.7
                except Exception:
                    kwargs["temperature"] = 0.7
            else:
                kwargs["temperature"] = 0.7
        return kwargs

    @staticmethod
    def _messages(system: str, user: str) -> List[MessageParam]:
        return [
            ChatCompletionSystemMessageParam(role="system", content=system),
            ChatCompletionUserMessageParam(role="user", content=user),
        ]

    @staticmethod
    def _chunk_segments(segments: List[dict], chunk_size: int) -> List[List[dict]]:
        chunk_size = max(1, chunk_size)
        return [
            segments[i : i + chunk_size]
            for i in range(0, len(segments), chunk_size)
        ]

    @staticmethod
    def _is_credit_exhausted_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "credit balance" in message or "insufficient credit" in message

    def _completion_typed(
        self,
        *,
        response_model: Type[T],
        messages: MessageList,
        enable_reasoning: bool = False,
    ) -> T:
        kwargs = self._build_kwargs(
            response_model=response_model,
            messages=messages,
            enable_reasoning=enable_reasoning,
        )
        try:
            return self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self._client.provider is Provider.ANTHROPIC and self._is_credit_exhausted_error(exc):
                fallback_key = os.getenv("OPENAI_API_KEY")
                if fallback_key:
                    fallback_model = (os.getenv("LLM_FALLBACK_MODEL") or "gpt-5-mini").strip()
                    logger.warning(
                        "Anthropic credits exhausted; falling back to OpenAI provider with model '%s'.",
                        fallback_model,
                    )
                    self._client = create_instructor_client(
                        provider_override=Provider.OPENAI,
                        model_override=fallback_model,
                    )
                    kwargs = self._build_kwargs(
                        response_model=response_model,
                        messages=messages,
                        enable_reasoning=enable_reasoning,
                    )
                    return self._client.chat.completions.create(**kwargs)
                raise RuntimeError(
                    "Anthropic API retornou créditos insuficientes e nenhuma chave OPENAI_API_KEY está configurada. "
                    "Renove os créditos da Anthropic ou defina OPENAI_API_KEY (opcionalmente LLM_FALLBACK_MODEL) para habilitar o fallback automático."
                ) from exc
            raise

    @staticmethod
    def _cap_assets_for_duration(duration: float, requested_count: int) -> int:
        """Clamp the number of assets to a reasonable maximum for the scene duration."""
        safe_duration = max(0.1, duration)
        if safe_duration < 3.0:
            max_assets = 1
        elif safe_duration < 5.5:
            max_assets = 2
        elif safe_duration < 8.0:
            max_assets = 3
        else:
            max_assets = 4
        return max(1, min(requested_count, max_assets))

    def _generate_recipe_metadata(
        self,
        *,
        topic: str,
        narration_text: str,
        total_duration: float,
        audio_path: str,
        video_profile: VideoProfile | None,
    ) -> VideoRecipeMetadata:
        narration_preview = " ".join(narration_text.strip().split())
        if len(narration_preview) > 800:
            narration_preview = narration_preview[:800].rstrip() + "…"

        render_fps = video_profile.fps if video_profile else 60
        if video_profile and video_profile.orientation == "portrait":
            orientation_label = (
                f"vertical {video_profile.width}x{video_profile.height} (aspect_ratio {video_profile.aspect_ratio})"
            )
        elif video_profile:
            orientation_label = (
                f"horizontal {video_profile.width}x{video_profile.height} (aspect_ratio {video_profile.aspect_ratio})"
            )
        else:
            orientation_label = ""

        orientation_hint = (
            f"- Formato visual alvo: {orientation_label}.\n" if orientation_label else ""
        )

        system = (
            "Você é um especialista VIRAL em YouTube Shorts/TikTok com 10+ anos otimizando algoritmos.\n"
            f"{BRAND_CONTEXT}\n\n"
            "OBJETIVO: Criar metadados que GARANTAM CTR 15%+ e AVD 85%+ através de gatilhos psicológicos comprovados.\n\n"
            "🎯 FÓRMULA DO TÍTULO VIRAL (40-60 chars):\n\n"
            "HOOKS TESTADOS E COMPROVADOS (CTR 20%+):\n"
            "• Pergunta Direta + Emoji: 'Por que você [problema comum]? 😰'\n"
            "• Contradição + Emoji: '🤯 [Crença comum] está ERRADO'\n"
            "• Curiosity Gap + Emoji: '⚠️ O que [fenômeno] revela sobre você'\n"
            "• Identificação + Emoji: '😱 Se você [ação], isso acontece'\n"
            "• Revelação + Emoji: '🔥 O que [especialistas] escondem sobre [X]'\n"
            "• Dark Hook + Emoji: '☠️ A verdade sobre [X] que ninguém conta'\n\n"
            "REGRAS CRÍTICAS:\n"
            "- SEMPRE começar com pergunta direta OU emoji relevante (não ambos obrigatoriamente)\n"
            "- Use 1-2 palavras em CAPS para ênfase estratégica\n"
            "- Números específicos quando aplicável: '5x mais chance' > 'muito mais'\n"
            "- Criar identificação imediata: 'você', 'seu cérebro', 'acontece com você'\n"
            "- NUNCA use: 'Você sabia que...', 'Curiosidade sobre...', 'Fato interessante...', 'ALARME', 'PERSEGUE'\n"
            "- Evite verbos genéricos tipo 'alarme', 'persegue' - seja específico e direto\n\n"
            "📝 DESCRIÇÃO OTIMIZADA (100-150 palavras):\n\n"
            "ESTRUTURA COMPROVADA:\n"
            "[Linha 1]: Hook direto - afirmação forte sobre o tema\n"
            "[Linha 2]: Linha em branco para respiração visual\n"
            "[Linha 3-4]: Explicação científica/factual em linguagem acessível\n"
            "[Linha 5]: Linha em branco\n"
            "[Linha 6-7]: Revelação impactante - 'O pior?' ou dados surpreendentes\n"
            "[Linha 8]: Linha em branco\n"
            "[Linha 9-10]: Teaser com números/estudos específicos\n"
            "[Linha 11]: Linha em branco\n"
            "[Linha 12]: CTA com emoji - 'Fica até o final' + promessa de valor\n"
            "[Linha 13]: Linha em branco\n"
            "[Linha 14]: CTA social - 'Marca aquela pessoa' com emoji 👇\n"
            "[Linha 15]: Linha em branco\n"
            "[Linha 16]: 5-7 hashtags relevantes\n\n"
            "REGRAS DE ESCRITA:\n"
            "- Parágrafos curtos (1-2 linhas cada) com LINHAS EM BRANCO entre eles\n"
            "- Use CAPS apenas em 2-3 palavras-chave para impacto\n"
            "- Linguagem conversacional e direta\n"
            "- Dados específicos > afirmações vagas\n"
            "- Tom: 'fascinante mas acessível'\n\n"
            "HASHTAGS ESTRATÉGICAS (5-7 no final da descrição):\n"
            "PRIORIZAÇÃO:\n"
            "1. #Shorts (sempre primeira)\n"
            "2. Hashtag PRINCIPAL do tema (ex: #AcordarAs3h)\n"
            "3. 2-3 hashtags do nicho (ex: #Insônia #Ansiedade #Cortisol)\n"
            "4. 2-3 hashtags amplas (ex: #SaúdeMental #CiênciaDoSono)\n"
            "Evite: #CuriosidadesBR #FatosCuriosos (muito genéricas)\n\n"
            "🏷️ TAGS ESTRATÉGICAS (12-15 tags):\n"
            "MIX OBRIGATÓRIO:\n"
            "• 2-3 tags longtail EXATAS do tema: frases que pessoas buscam\n"
            "  Exemplo: 'acordar às 3 da manhã', 'por que acordo às 3h'\n"
            "• 3-4 tags específicas do nicho: palavras-chave principais\n"
            "  Exemplo: 'insônia', 'ansiedade', 'cortisol', 'sono'\n"
            "• 3-4 tags médias relacionadas: contexto do tema\n"
            "  Exemplo: 'saúde mental', 'ciência do sono', 'acordar de madrugada'\n"
            "• 2-3 tags amplas: alcance geral\n"
            "  Exemplo: 'shorts brasil', 'curiosidades', 'fatos científicos'\n\n"
            "PRIORIZAÇÃO DE TAGS:\n"
            "- As 3-4 primeiras devem ser as mais específicas (longtail)\n"
            "- Tags devem refletir termos REAIS de busca\n"
            "- Evite tags muito genéricas tipo 'brasil', 'vídeo curto'\n"
            "- Inclua variações naturais: 'acordar às 3h' E 'acordar de madrugada'\n\n"
            "⚠️ REGRAS DE SEGURANÇA YOUTUBE:\n"
            "✅ PERMITIDO: Fatos científicos educativos, fenômenos naturais, curiosidades psicológicas\n"
            "❌ PROIBIDO: Clickbait enganoso, sensacionalismo vazio, promessas não cumpridas\n\n"
            "PSICOLOGIA DO CONTEÚDO:\n"
            "- Curiosidade informativa (não sensacionalismo)\n"
            "- Identificação pessoal ('acontece com você')\n"
            "- Valor educativo real + entretenimento\n"
            "- Tom: 'isso é fascinante E você vai aprender algo'\n\n"
            "FORMATO: JSON `VideoRecipeMetadata` com todos campos otimizados para máximo alcance viral."
        )

        user = (
            f"TEMA: {topic.strip()}\n"
            f"CANAL: {CHANNEL_BRAND}\n\n"
            f"NARRAÇÃO (preview):\n{narration_preview}\n\n"
            "ESPECIFICAÇÕES TÉCNICAS:\n"
            f"- audio.path: {audio_path}\n"
            f"- audio.duration_sec: {total_duration:.3f}\n"
            f"- policy.render: adapter='editly', fps={render_fps}, audio_codec='aac', pixel_format='yuv420p', preset='ultrafast'\n"
            "- fade_in_sec: 0.2\n"
            "- fade_out_sec: 0.3\n"
            f"{orientation_hint}\n"
            "- language: 'pt-BR'\n\n"
            "CRIE METADADOS QUE:\n"
            "1. GARANTAM clique imediato (CTR 15%+)\n"
            "2. Usem gatilhos psicológicos dark mas YouTube-safe\n"
            "3. Prometam valor que o vídeo REALMENTE entrega\n"
            "4. Otimizem para busca E algoritmo simultaneamente\n"
            "5. Gerem compartilhamento orgânico ('você precisa ver isso')\n\n"
            "Retorne JSON válido com metadados VIRAIS."
        )

        messages: List[MessageParam] = self._messages(system, user)
        metadata = self._completion_typed(
            response_model=VideoRecipeMetadata,
            messages=messages,
        )
        metadata.language = "pt-BR"
        metadata.audio.path = audio_path
        metadata.audio.duration_sec = round(total_duration, 3)
        if video_profile:
            try:
                metadata.policy.render.fps = video_profile.fps
            except Exception:
                pass
        return metadata

    def generate_music_prompt(
        self,
        topic: str,
        narration_text: str,
        duration_sec: float,
    ) -> MusicPlan:
        narration_compact = " ".join((narration_text or "").strip().split())
        if len(narration_compact) > 600:
            narration_compact = narration_compact[:600].rstrip() + "…"

        system = (
            "Você é produtor musical especializado em trilhas VIRAIS para YouTube Shorts/TikTok.\n"
            f"{BRAND_CONTEXT}\n\n"
            "MISSÃO: Criar prompt para ElevenLabs Music que gere trilha HIPNOTIZANTE para máxima retenção.\n\n"
            "🎵 FÓRMULA DA TRILHA VIRAL (pesquisas 2024-2025):\n\n"
            "ESTRUTURA COMPROVADA (85%+ completion rate):\n"
            "1. INÍCIO IMEDIATO (0-3s): Sem intro, direto no groove principal\n"
            "2. BUILD SUTIL (3-20s): Tensão crescente mas NUNCA drop pesado\n"
            "3. SUSTENTAÇÃO (20-40s): Energia constante com micro-variações\n"
            "4. LOOP PERFEITO (40-45s): Final conecta suavemente ao início\n\n"
            "ELEMENTOS OBRIGATÓRIOS NO PROMPT:\n"
            "• Ritmo/Pulse: SEMPRE incluir 'steady pulse', 'constant rhythm', 'hypnotic beat'\n"
            "• Início: SEMPRE incluir 'start immediately with main groove, no intro'\n"
            "• Background: SEMPRE incluir 'sits softly behind voice', 'supports narration'\n"
            "• Atmosfera: Usar palavras emocionais, não técnicas\n\n"
            "VOCABULÁRIO PARA DARK/CURIOSIDADES:\n"
            "✅ USAR: mysterious, eerie, tense, dark ambient, unsettling undertone, psychological tension\n"
            "✅ USAR: curious, discovery, scientific wonder, space-like, deep ocean vibes\n"
            "✅ USAR: cinematic, atmospheric, moody, haunting but not horror\n"
            "❌ EVITAR: BPM, Hz, dB, seconds, duration, loud, aggressive, horror\n\n"
            "INSTRUMENTAÇÃO VIRAL:\n"
            "Dark content: deep bass, dark synth pads, subtle strings, atmospheric drones\n"
            "Curiosidades: soft bells, ethereal pads, light percussion, cosmic synths\n"
            "Tensão: rising strings, ticking clock, heartbeat rhythm, breath sounds\n\n"
            "ESTRUTURA DO PROMPT (100-250 chars):\n"
            "[Mood] + [Rhythm] + [Instruments] + [Atmosphere] + [Mixing]\n\n"
            "EXEMPLOS DE ALTA PERFORMANCE:\n"
            "'Dark mysterious atmosphere with steady hypnotic beat, deep bass and ethereal pads, "
            "psychological tension building, sits softly behind voice, start immediately'\n\n"
            "'Eerie scientific discovery vibe, constant subtle pulse, atmospheric drones and soft bells, "
            "unsettling but curious, supports narration without overpowering'\n\n"
            "FORMATO: JSON `MusicPlan` com campo `prompt` em INGLÊS."
        )

        user = (
            f"TEMA: {topic.strip()}\n\n"
            f"NARRAÇÃO (resumo):\n{narration_compact}\n\n"
            f"DURAÇÃO: {duration_sec:.2f}s\n\n"
            "Crie prompt (100-250 chars) para trilha que:\n"
            "1. HIPNOTIZE o espectador (ritmo constante viciante)\n"
            "2. Amplifique emoção dark/curiosa sem dominar voz\n"
            "3. Comece IMEDIATAMENTE no groove principal\n"
            "4. Crie tensão psicológica sutil\n"
            "5. Loop perfeito para rewatches\n\n"
            "NÃO mencione duração, BPM ou termos técnicos."
        )

        messages: List[MessageParam] = self._messages(system, user)
        return self._completion_typed(
            response_model=MusicPlan,
            messages=messages,
        )

    def generate_segment_grouping_plan(
        self,
        *,
        segments: List[dict],
        topic: str,
    ) -> SegmentGroupingPlan:
        segments_summary = []
        for seg in segments:
            idx = seg.get("index", 0)
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = (seg.get("text", "") or "").strip()
            duration = round(end - start, 2)
            segments_summary.append(f"{idx}. [{start:.1f}-{end:.1f}s, {duration}s] \"{text}\"")

        segments_text = "\n".join(segments_summary)
        total_segments = len(segments)

        system = (
            "Você é especialista em ritmo narrativo VIRAL para YouTube Shorts/TikTok.\n"
            f"{BRAND_CONTEXT}\n\n"
            "MISSÃO: Agrupar segments para criar ritmo HIPNOTIZANTE com mudanças visuais a cada 2.5-3.5s.\n\n"
            "⚡ CIÊNCIA DO RITMO VIRAL (dados 2024-2025):\n"
            "- Mudança visual a cada 2.5-3.5s = 35% mais retenção\n"
            "- Grupos de 3-5s = ritmo ideal para não cansar\n"
            "- Variação de duração mantém atenção\n\n"
            "REGRAS DE AGRUPAMENTO VIRAL:\n\n"
            "1) SEMPRE AGRUPAR (obrigatório para fluxo):\n"
            "   A) Fragmentos < 2s: Muito curtos, quebram ritmo\n"
            "   B) Interjeições isoladas: 'Sério?', 'Nossa!', 'Olha só'\n"
            "   C) Conectivos: 'Mas tem mais', 'E olha', 'Agora vem'\n"
            "   D) Builds de tensão: frases que amplificam mistério\n\n"
            "2) MANTER SEPARADO (para impacto):\n"
            "   A) Revelações importantes (payoff moments)\n"
            "   B) Mudanças de conceito visual\n"
            "   C) Dados/números impactantes\n"
            "   D) Hooks e cliff-hangers\n\n"
            "3) DURAÇÕES IDEAIS POR GRUPO:\n"
            "   - MÍNIMO: 2.5s (menos que isso = corte frenético)\n"
            "   - IDEAL: 3.0-5.0s (1-2 assets por cena)\n"
            "   - MÁXIMO: 6.0s (acima = muito longo, perde ritmo)\n\n"
            "4) ESTRATÉGIA DE TENSÃO:\n"
            "   - Início (0-15s): Grupos de 3-4s (estabelece ritmo)\n"
            "   - Meio (15-35s): Varie 2.5-5s (mantém interesse)\n"
            "   - Clímax (35-45s): Grupos de 3-4s (acelera para final)\n\n"
            "5) PROCESSO DE ANÁLISE (USE REASONING):\n"
            "   Para cada segment, analise:\n"
            "   - É hook/payoff? → Manter isolado para impacto\n"
            "   - É build/contexto? → Agrupar para fluxo\n"
            "   - Tem dado chocante? → Isolar para ênfase\n"
            "   - É transição? → Agrupar com próximo\n\n"
            "FORMATO: JSON `SegmentGroupingPlan` com grupos otimizados para ritmo viral."
        )

        user = (
            f"TEMA: {topic.strip()}\n"
            f"TOTAL DE SEGMENTS: {total_segments}\n\n"
            f"SEGMENTS COMPLETOS:\n{segments_text}\n\n"
            "ANALISE E AGRUPE para:\n"
            "1. Criar ritmo HIPNOTIZANTE (mudanças a cada 2.5-3.5s)\n"
            "2. Preservar momentos de IMPACTO (hooks, payoffs)\n"
            "3. Manter FLUXO narrativo natural\n"
            "4. Variar durações para evitar monotonia\n"
            "5. Otimizar para edição com cortes no ritmo\n\n"
            "USE REASONING para decisões criteriosas sobre cada agrupamento.\n"
            "Retorne APENAS índices dos grupos, sem prompts visuais."
        )

        messages: List[MessageParam] = self._messages(system, user)
        return self._completion_typed(
            response_model=SegmentGroupingPlan,
            messages=messages,
            enable_reasoning=True,
        )

    def generate_segments_from_transcription(
        self,
        *,
        full_transcription: str,
        audio_duration_sec: float,
        topic: str,
    ):
        transcription_compact = " ".join((full_transcription or "").strip().split())
        if len(transcription_compact) > 2000:
            transcription_compact = transcription_compact[:2000].rstrip() + "…"

        system = (
            "Você é especialista em segmentação narrativa VIRAL para YouTube Shorts/TikTok.\n"
            f"{BRAND_CONTEXT}\n\n"
            "MISSÃO: Criar segmentos que FORCEM mudanças visuais a cada 2.5-3.5s para máxima retenção.\n\n"
            "📊 CIÊNCIA DA SEGMENTAÇÃO VIRAL:\n"
            "Pesquisas mostram que Shorts com cortes a cada 2.5-3.5s têm:\n"
            "• 35% mais taxa de conclusão\n"
            "• 60% mais rewatches\n"
            "• 45% mais compartilhamentos\n\n"
            "DURAÇÕES IDEAIS POR TIPO:\n"
            "• Hook inicial: 2.5-3.5s (impacto imediato)\n"
            "• Builds de tensão: 3.0-4.0s (sustenta interesse)\n"
            "• Revelações: 2.0-3.0s (momento 'wow')\n"
            "• Dados/números: 1.5-2.5s (digestão rápida)\n"
            "• Payoff final: 3.0-4.0s (satisfação)\n\n"
            "QUEBRAS ESTRATÉGICAS (ordem de prioridade):\n"
            "1. APÓS hooks/cliff-hangers (maximiza curiosidade)\n"
            "2. ANTES de revelações (cria antecipação)\n"
            "3. Em mudanças de tópico/conceito\n"
            "4. Após perguntas retóricas\n"
            "5. Em pausas dramáticas naturais\n\n"
            "NUNCA QUEBRAR:\n"
            "• No meio de dados importantes\n"
            "• Durante builds de tensão\n"
            "• Em frases de impacto emocional\n\n"
            "REGRAS DE UNIÃO:\n"
            "• Interjeições SEMPRE com contexto: 'destruindo. Sério? Até os oceanos'\n"
            "• Conectivos SEMPRE com continuação: 'E tem mais. A atmosfera...'\n"
            "• Números PODEM ficar isolados se impactantes: '1.673 km por hora.'\n\n"
            "DISTRIBUIÇÃO TEMPORAL:\n"
            "• 0-15s: Segmentos de 2.5-3.5s (estabelece ritmo rápido)\n"
            "• 15-35s: Varie entre 2.0-4.0s (evita previsibilidade)\n"
            "• 35-45s: Segmentos de 3.0-4.0s (permite absorção do payoff)\n\n"
            "VALIDAÇÕES CRÍTICAS:\n"
            "✓ Duração mínima: 1.5s (legibilidade de legenda)\n"
            "✓ Duração máxima: 5.0s (evita monotonia)\n"
            "✓ Média ideal: 2.5-3.5s por segmento\n"
            "✓ Variação obrigatória (não repetir mesma duração 3x seguidas)\n\n"
            "FORMATO: JSON `SegmentsPlan` com segments otimizados para edição viral."
        )

        user = (
            f"TEMA: {topic.strip()}\n"
            f"DURAÇÃO TOTAL: {audio_duration_sec:.2f}s\n\n"
            f"TRANSCRIÇÃO:\n{transcription_compact}\n\n"
            "SEGMENTE para:\n"
            "1. FORÇAR mudanças visuais a cada 2.5-3.5s\n"
            "2. Criar RITMO hipnotizante mas variado\n"
            "3. Preservar IMPACTO de hooks e revelações\n"
            "4. Facilitar edição com CORTES naturais\n"
            "5. Distribuir timestamps PROPORCIONALMENTE\n\n"
            "USE REASONING para criar segmentação PERFEITA para viral.\n"
            f"Último segment DEVE terminar em {audio_duration_sec:.2f}s (±0.1s)."
        )

        messages: List[MessageParam] = self._messages(system, user)
        return self._completion_typed(
            response_model=SegmentsPlan,
            messages=messages,
            enable_reasoning=True,
        )

    def _generate_scene_blueprint(
        self,
        *,
        topic: str,
        narration_chunk: str,
        segment: dict,
        scene_index: int,
        total_scenes: int,
    ) -> SceneBlueprint:
        segment_text = str(segment.get("text", "")).strip()
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", seg_start))
        segment_duration = max(0.1, seg_end - seg_start)

        system = (
            "Você é diretor visual viral de vídeos dark para YouTube Shorts e TikTok.\n"
            f"{BRAND_CONTEXT}\n"
            "Gere JSON `SceneBlueprint` completo, consistente e cinematográfico."
        )

        user = (
            f"TEMA: {topic.strip()}\n"
            f"CENA: {scene_index} de {total_scenes}\n"
            f"SEGMENTO: {segment.get('index', scene_index + 1)}\n"
            f"DURAÇÃO DO SEGMENTO: {segment_duration:.3f}s\n"
            f"TEXTO DO SEGMENTO:\n{segment_text}\n\n"
            f"NARRAÇÃO DO BLOCO:\n{narration_chunk.strip()}\n\n"
            "Defina scene_role, visual_mode, intent, emotion, motion_style, color_mood, asset_count, impact_level, overlay_text "
            "e demais campos do blueprint com total autonomia criativa.\n"
            "Regras de asset_count por duração: <3.0s = 1 asset; 3.0-5.5s = 2 assets; 5.5-8.0s = 3 assets; >8.0s = 4 assets (máximo)."
        )

        messages: List[MessageParam] = self._messages(system, user)
        return self._completion_typed(
            response_model=SceneBlueprint,
            messages=messages,
        )

    def _generate_assets_from_blueprint(
        self,
        *,
        topic: str,
        blueprint: SceneBlueprint,
        segment: dict,
        video_profile: VideoProfile | None,
    ) -> List[Asset]:
        segment_text = str(segment.get("text", "")).strip()
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", seg_start))
        segment_duration = max(0.1, seg_end - seg_start)
        blueprint_payload = json.dumps(blueprint.model_dump(), ensure_ascii=False)
        profile_payload = json.dumps(video_profile.to_dict() if video_profile else None, ensure_ascii=False)

        system = (
            "Você cria a lista de `Asset` para a cena seguindo o schema do projeto.\n"
            f"{BRAND_CONTEXT}\n"
            "Regras obrigatórias:\n"
            "- semantic_text e semantic_text_variants em inglês com 8-12 palavras.\n"
            "- negative_semantic_texts com 3-6 itens claros.\n"
            "- search_queries concisas (até 5 palavras) e com pelo menos 3 opções distintas.\n"
            "- generate_prompt em inglês, cinematográfico, mínimo 100 caracteres; video_generate_prompt obrigatório quando type='video'.\n"
            "- duration_hint_sec deve dividir a duração do segmento pelo asset_count do blueprint.\n"
            "- Todos os assets type='image' precisam de zoomDirection e zoomAmount.\n"
            "- Transitions devem ser válidas e coerentes.\n"
            "- Respeite as regras de search_strategy já usadas no projeto."
        )

        user = (
            f"TEMA: {topic.strip()}\n"
            f"BLUEPRINT:\n{blueprint_payload}\n\n"
            f"SEGMENTO #{segment.get('index', blueprint.scene_index)} ({segment_duration:.3f}s):\n{segment_text}\n\n"
            f"VIDEO_PROFILE:\n{profile_payload}\n\n"
            f"Gere exatamente {blueprint.asset_count} assets completos, com prompts, durations, transitions, motion e metadados sem depender de lógica externa."
        )

        messages: List[MessageParam] = self._messages(system, user)
        return self._completion_typed(
            response_model=List[Asset],
            messages=messages,
        )

    def _generate_recipe_scene_chunk(
        self,
        *,
        topic: str,
        narration_chunk: str,
        segments_chunk: List[dict],
        scene_index_start: int,
        total_scenes: int,
        video_profile: VideoProfile | None,
    ) -> VideoRecipeSceneChunk:
        if not segments_chunk:
            return VideoRecipeSceneChunk()

        scenes: List[Scene] = []
        total_scene_count = max(total_scenes, len(segments_chunk))

        for offset, segment in enumerate(segments_chunk):
            blueprint = self._generate_scene_blueprint(
                topic=topic,
                narration_chunk=narration_chunk,
                segment=segment,
                scene_index=scene_index_start + offset,
                total_scenes=total_scene_count,
            )

            start_time = float(segment.get("start", 0.0))
            end_time = float(segment.get("end", start_time))
            segment_duration = max(0.1, end_time - start_time)

            capped_assets = self._cap_assets_for_duration(segment_duration, blueprint.asset_count)
            if capped_assets != blueprint.asset_count:
                blueprint = blueprint.model_copy(update={"asset_count": capped_assets})

            assets = self._generate_assets_from_blueprint(
                topic=topic,
                blueprint=blueprint,
                segment=segment,
                video_profile=video_profile,
            )

            scene = Scene(
                index=blueprint.scene_index,
                start_time=round(start_time, 3),
                end_time=round(end_time, 3),
                text=str(segment.get("text", "")),
                overlay_text=blueprint.overlay_text,
                visual_mode=blueprint.visual_mode,
                intent=blueprint.intent,
                transition=assets[0].transition if assets else "",
                assets=assets,
            )

            scenes.append(scene)

        return VideoRecipeSceneChunk(scenes=scenes)

    def _generate_base_narrative_text(self, topic: str) -> str:
        target_secs = str(os.getenv("NARRATION_TARGET_SECS", "45s"))
        system = (
            "Você é o MELHOR roteirista viral do YouTube Shorts Brasil com 50M+ views mensais.\n"
            f"{BRAND_CONTEXT}\n\n"
            "O canal/quadro se chama 'Isso tem explicação'. A PROMESSA central é: "
            "o VÍDEO em si entrega a explicação completa do fenômeno, sem depender de comentários ou vídeos futuros.\n\n"
            "🎯 MISSÃO: Criar roteiro que GARANTA 90%+ completion rate e 150%+ watch time (rewatches).\n\n"
            "⚡ ESTRUTURA W.A.V.E. COMPROVADA (dados 2024-2025):\n\n"
            "📱 [0-3s] HOOK DEVASTADOR (60% do sucesso está aqui):\n"
            "FÓRMULAS QUE CONVERTEM 20%+ CTR:\n"
            "• Consequência Direta: 'Se [X] parasse agora, você morreria em [tempo exato]'\n"
            "• Contradição Chocante: 'Tudo que te ensinaram sobre [X] é mentira'\n"
            "• Dark Fact: 'Existe algo em [lugar comum] que pode [consequência terrível]'\n"
            "• Número Impossível: '[Número absurdo] [unidade] de [coisa inesperada]'\n"
            "• Pergunta Perturbadora: 'Você sabe por que [fenômeno comum] é [dark fact]?'\n\n"
            "🌊 [3-15s] AGITAÇÃO - Amplificação do Problema:\n"
            "• Adicione camada mais perturbadora: 'Mas isso não é nem a pior parte...'\n"
            "• Dados específicos que chocam: números exatos, comparações viscerais\n"
            "• Crie micro-loops: 'E sabe o que acontece depois?'\n"
            "• Tags ElevenLabs: [curious], [dramatic] para tensão\n\n"
            "🔥 [15-35s] REVELAÇÃO PROGRESSIVA com Loops:\n"
            "• Entregue informação em CAMADAS (não tudo de uma vez)\n"
            "• Cada revelação gera nova pergunta\n"
            "• Use: 'Cientistas descobriram...', 'Estudos mostram...', 'O que ninguém te conta...'\n"
            "• Alternância: fato → reação → novo fato → amplificação\n"
            "• Tags: [whisper] para segredos, [excited] para descobertas\n\n"
            "💥 [35-42s] PAYOFF + TWIST:\n"
            "• Entregue a promessa do hook COMPLETAMENTE dentro deste vídeo\n"
            "• Adicione twist final: 'Mas tem um detalhe...'\n"
            "• Conecte ao cotidiano: 'Isso significa que você...'\n"
            "• Tag [serious] para impacto final\n\n"
            "🔄 [42-45s] LOOP DE REWATCH:\n"
            "• Conecte final ao início: 'E é por isso que [referência ao hook]'\n"
            "• Ou abra novo mistério RELACIONADO: 'Se isso já é estranho, imagina quando você descobrir [próximo fato]'\n"
            "• CTA viciante, mas SEM prometer explicações futuras: use coisas como\n"
            "  - 'Comenta se isso já aconteceu com você'\n"
            "  - 'Marca alguém que precisa ver isso'\n"
            "  - 'Se isso fez sentido, segue o canal pra não perder a próxima explicação'\n\n"
            "📏 FORMATO DO ROTEIRO (45s):\n"
            "• Quebre o texto em 6-8 blocos separados por linhas em branco, cada um com 1-2 frases curtas.\n"
            "• Bloco 1 = Hook devastador. Blocos 2-3 = Agitação. Blocos 4-5 = Revelações em camadas. Bloco 6 = Payoff + Twist. Bloco final = Loop que conecta ao início.\n"
            "• Use conectores que criem micro-loops: 'E olha o pior...', 'Saca só o detalhe...'.\n"
            "• Mire em 100-115 palavras totais (ritmo 135-150 wpm para caber em 45s).\n"
            "• Prefira frases de 10-14 palavras para manter fôlego sem estourar tempo.\n"
            "• Evite formato de lista; é narrativa falada.\n\n"
            "🌑 ATMOSFERA DARK E HIPNÓTICA:\n"
            "• Puxe o espectador para um clima de tensão controlada: descreva sombras, sons abafados, texturas estranhas.\n"
            "• Alimente curiosidade com frases que pareçam quase proibidas ('ninguém comenta, mas...').\n"
            "• Use contrastes: cotidiano confortável vs. consequência perturbadora.\n"
            "• Nunca caia no gore; mantenha suspense psicológico.\n"
            "• Mantenha expectativa ativa citando o que pode dar errado se ignorarem o fenômeno.\n\n"
            "🌀 FLUXO SENSORIAL → EXPLICAÇÃO → IMPACTO:\n"
            "• COMEÇO: descreva o que o corpo ou ambiente sente (barulho oco, cheiro de madeira, arrepio na nuca, luz piscando, etc.).\n"
            "• LOGO EM SEGUIDA: explique o mecanismo em linguagem simples (\"é porque o tronco cerebral...\", \"isso rola porque a madeira seca mais rápido...\").\n"
            "• IMPACTO: traduza em uma frase como isso afeta a vida do espectador (segurança, bolso, saúde, família).\n"
            "• CTA/eco social fecha o bloco com pergunta ou comando.\n"
            "• Máximo de 2 números/dados por roteiro inteiro; use só quando realmente surpreender.\n\n"
            "📌 NÃO REPITA O ÓBVIO:\n"
            "• Considere que o público já ouviu falar do fenômeno; traga ângulos novos, bastidores, consequências invisíveis.\n"
            "• Use comparação para entregar novidade (\"no Brasil a gente... enquanto lá fora...\").\n"
            "• Se precisar contextualizar, faça em 1 frase antes da nova informação.\n\n"
            "🌍 CONTEXTO LOCAL QUANDO RELEVANTE:\n"
            "• Mostre como o fenômeno toca a vida de quem tá assistindo, mas só destaque comparações culturais quando elas realmente ajudam (evite frases repetidas como 'no Brasil...').\n"
            "• Se o fenômeno for estrangeiro, explique em 1 frase qual seria o paralelo mais próximo aqui, sem insistir nisso o roteiro inteiro.\n\n"
            "👥 PÚBLICO 25-65+:\n"
            "• Linguagem madura sem perder ritmo: misture ganchos fortes com vocabulário acessível pra quem já viveu muita coisa.\n"
            "• Use referências de cotidiano adulto (trabalho, família, finanças, saúde) quando fizer sentido.\n"
            "• Mostre utilidade imediata, legado ou proteção da família para aumentar engajamento.\n\n"
            "MODELOS DE HOOK QUE ESTÃO PERFORMANDO:\n"
            "• 'Já acordou [sensação bizarra]? Isso é [fenômeno] e X% das pessoas...'\n"
            "• 'Seu corpo tá [reação]? Culpa de [processo escondido] que tá ativo agora.'\n"
            "• 'Se você [hábito comum], seu cérebro [castigo/recompensa] em [número específico].'\n\n"
            "🧪 JARGÃO? SÓ COM TRADUÇÃO IMEDIATA:\n"
            "• Sempre descreva a sensação ou metáfora cotidiana antes do termo técnico.\n"
            "• Ao citar neurotransmissores, hormônios ou estruturas, explique em uma frase curta o que fazem (ex: 'um freio químico que desliga teus músculos — neurotransmissores calmantes').\n"
            "• Prefira termos simples; se precisar usar o nome técnico, encaixe como curiosidade complementar ('os neurologistas chamam isso de...').\n"
            "• Evite listas frias de termos; transforme cada conceito em imagem mental.\n"
            "• Se o nome não for necessário, prefira só o efeito percebido pelo espectador.\n\n"
            "🇧🇷 LINGUAGEM BRASILEIRA VIRAL:\n"
            "NATURALMENTE use (sem forçar):\n"
            "• Contrações: 'tá', 'cê', 'pra' (máximo naturalidade)\n"
            "• Expressões: 'cara', 'olha só', 'pois é' (1-2 por vídeo)\n"
            "• Gírias: dose certa, não exagere\n"
            "• Tom: como se explicasse para um amigo curioso, não como professor\n\n"
            "🏷️ TAGS ELEVENLABS ESTRATÉGICAS:\n"
            "[whisper] - momentos de revelação perturbadora\n"
            "[dramatic] - builds de tensão\n"
            "[curious] - perguntas que geram curiosidade\n"
            "[excited] - descobertas fascinantes\n"
            "[serious] - fatos graves ou conclusões\n"
            "[pause] - antes de revelações (USE MUITO)\n"
            "[fast] - listas rápidas, urgência\n"
            "[slow] - ênfase em números ou fatos chocantes\n"
            "As tags SEMPRE devem permanecer em inglês exatamente como listado (nunca traduza para pt-BR).\n\n"
            "⚠️ TÉCNICAS PSICOLÓGICAS DARK:\n"
            "• Efeito Zeigarnik: deixe questões abertas, mas SEM quebrar a promessa de explicar o fenômeno principal\n"
            "• Curiosity Gap: tensão entre conhecido e desconhecido\n"
            "• Fear Appeal controlado: medo produtivo, não pânico\n"
            "• Fascínio mórbido: explorar o que não deveria interessar\n"
            "• Contraste extremo: cotidiano vs extraordinário\n\n"
            "📊 MÉTRICAS QUE IMPORTAM:\n"
            "• Palavras por minuto: 140-160 (ritmo brasileiro natural)\n"
            "• Frases curtas: 10-15 palavras (digestão fácil)\n"
            "• Hooks secundários: a cada 10-15s (mantém atenção)\n"
            "• Pausas estratégicas: 3-5 por vídeo (impacto)\n\n"
            "🚫 EVITE COMPLETAMENTE:\n"
            "• 'Você sabia que...' (clichê morto)\n"
            "• 'Olá pessoal' (mata retenção)\n"
            "• Enrolação ou contexto desnecessário\n"
            "• Promessas não cumpridas\n"
            "• Tom professoral ou condescendente\n"
            "• Qualquer frase que indique explicação futura fora do vídeo, como:\n"
            "  - 'Comenta que eu te explico'\n"
            "  - 'Posso te explicar o que tá rolando'\n"
            "  - 'Te explico nos comentários'\n"
            "  - 'Te conto no próximo vídeo'\n\n"
            "✨ GATILHOS DE COMPARTILHAMENTO:\n"
            "• 'Ninguém acredita quando conto isso...'\n"
            "• 'Mostra pra quem duvida que...'\n"
            "• 'Marca aquele amigo que precisa saber...'\n"
            "• Informação que faz a pessoa parecer inteligente quando repete a história\n\n"
            "FORMATO: JSON `NarrativePlan` com `narration_text` VIRAL e VICIANTE, contendo a explicação completa do fenômeno dentro do próprio roteiro.\n"
            "REGRA EXTRA: Não insira tags do ElevenLabs ou qualquer anotação entre colchetes; retorne apenas texto limpo."
        )

        user = (
            f"TEMA: {topic.strip()}\n"
            f"DURAÇÃO: {target_secs}\n"
            f"CANAL: {CHANNEL_BRAND}\n\n"
            "Crie roteiro VIRAL que:\n"
            "1. CAPTURE em 3 segundos com hook IMPOSSÍVEL de ignorar\n"
            "2. Use estrutura W.A.V.E. para 90%+ completion\n"
            "3. Crie LOOPS que forcem rewatches (150%+ watch time)\n"
            "4. Explore psicologia DARK mas YouTube-safe\n"
            "5. Use linguagem brasileira NATURAL e viciante\n"
            "6. Integre tags ElevenLabs para jornada emocional\n"
            "7. Termine com CTA que gere engajamento REAL, mas SEM prometer explicações futuras ou ajuda individual\n"
            "8. Use o fluxo Sensorial → Explicação → Impacto nos blocos e só traga números quando forem realmente surpreendentes\n"
            "9. Evite repetir o que todo mundo já sabe; entregue ângulos novos ou comparações\n"
            "10. Só traga comparações culturais quando fizer sentido real para o espectador; foque no impacto direto\n"
            "11. Direcione o storytelling para quem tem entre 25 e 65+ anos, explorando dores e memórias dessa faixa\n"
            "12. Use linguagem simples ao tratar termos técnicos, explicando com metáforas antes de citar nomes científicos\n"
            "13. Sustente atmosfera dark e hipnótica com tensão psicológica, sem apelar para gore\n\n"
            "Regras IMPORTANTES de CTA e tom:\n"
            "• O VÍDEO deve conter a explicação completa do fenômeno. Não deixe a sensação de 'te explico depois'.\n"
            "• NÃO use frases como: 'posso te explicar o que tá rolando', 'comenta que eu te explico', "
            "'eu te explico nos comentários', 'te explico depois'.\n"
            "• Prefira CTAs de experiência/opinião: 'comenta se isso já aconteceu com você', 'marca alguém que precisa ver isso', "
            "'segue o canal se quer mais explicações assim'.\n\n"
            "LEMBRE: Os primeiros 3 segundos decidem TUDO.\n"
            "Cada frase deve ou entregar valor ou criar curiosidade.\n"
            "O final DEVE conectar ao início (loop perfeito) e reforçar que 'isso tem explicação' FOI mostrado neste vídeo."
        )

        messages: List[MessageParam] = self._messages(system, user)
        plan = self._completion_typed(
            response_model=NarrativePlan,
            messages=messages,
            enable_reasoning=True,
        )

        return plan.narration_text.strip()

    def _apply_elevenlabs_audio_tags(self, narration_text: str) -> str:
        clean_text = narration_text.strip()
        if not clean_text:
            return clean_text

        system = (
            "# Instructions\n\n"
            "## 1. Role and Goal\n\n"
            "You are an AI assistant specializing in enhancing dialogue text for speech generation.\n\n"
            "Your PRIMARY GOAL is to dynamically integrate audio tags (e.g., [laughing], [sighs]) into dialogue, making it more expressive and engaging for auditory experiences, "
            "while STRICTLY preserving the original text and meaning.\n\n"
            "It is imperative that you follow these system instructions to the fullest.\n\n"
            "## 2. Core Directives\n\n"
            "Follow these directives meticulously to ensure high-quality output.\n\n"
            "### Positive Imperatives (DO):\n\n"
            "* DO integrate audio tags from the \"Audio Tags\" list (or similar contextually appropriate audio tags) to add expression, emotion, and realism to the dialogue. "
            "These tags MUST describe something auditory.\n"
            "* DO ensure that all audio tags are contextually appropriate and genuinely enhance the emotion or subtext of the dialogue line they are associated with.\n"
            "* DO strive for a diverse range of emotional expressions (e.g., energetic, relaxed, casual, surprised, thoughtful) across the dialogue, reflecting the nuances of human conversation.\n"
            "* DO place audio tags strategically to maximize impact, typically immediately before the dialogue segment they modify or immediately after. "
            "(e.g., [annoyed] This is hard. or This is hard. [sighs]).\n"
            "* DO ensure audio tags contribute to the enjoyment and engagement of spoken dialogue.\n\n"
            "### Negative Imperatives (DO NOT):\n\n"
            "* DO NOT alter, add, or remove any words from the original dialogue text itself. Your role is to prepend audio tags, not to edit the speech. "
            "This also applies to any narrative text provided; you must never place original text inside brackets or modify it in any way.\n"
            "* DO NOT create audio tags from existing narrative descriptions. Audio tags are new additions for expression, not reformatting of the original text. "
            "(e.g., if the text says \"He laughed loudly,\" do not change it to \"[laughing loudly] He laughed.\" Instead, add a tag if appropriate, e.g., \"He laughed loudly [chuckles].\")\n"
            "* DO NOT use tags such as [standing], [grinning], [pacing], [music].\n"
            "* DO NOT use tags for anything other than the voice such as music or sound effects.\n"
            "* DO NOT invent new dialogue lines.\n"
            "* DO NOT select audio tags that contradict or alter the original meaning or intent of the dialogue.\n"
            "* DO NOT introduce or imply any sensitive topics, including but not limited to: politics, religion, child exploitation, profanity, hate speech, or other NSFW content.\n\n"
            "## 3. Workflow\n\n"
            "1. Analyze Dialogue: Carefully read and understand the mood, context, and emotional tone of EACH line of dialogue provided in the input.\n"
            "2. Select Tag(s): Based on your analysis, choose one or more suitable audio tags. Ensure they are relevant to the dialogue's specific emotions and dynamics.\n"
            "3. Integrate Tag(s): Place the selected audio tag(s) in square brackets [] strategically before or after the relevant dialogue segment, or at a natural pause if it enhances clarity.\n"
            "4. Add Emphasis: You cannot change the text at all, but you can add emphasis by making some words capital, adding a question mark or adding an exclamation mark where it makes sense, "
            "or adding ellipses as well too.\n"
            "5. Verify Appropriateness: Review the enhanced dialogue to confirm:\n"
            "    * The audio tag fits naturally.\n"
            "    * It enhances meaning without altering it.\n"
            "    * It adheres to all Core Directives.\n\n"
            "## 4. Output Format\n\n"
            "* Present ONLY the enhanced dialogue text in a conversational format.\n"
            "* Audio tags MUST be enclosed in square brackets (e.g., [laughing]).\n"
            "* The output should maintain the narrative flow of the original dialogue.\n\n"
            "## 5. Audio Tags (Non-Exhaustive)\n\n"
            "Use these as a guide. You can infer similar, contextually appropriate audio tags.\n\n"
            "Directions:\n"
            "* [happy]\n"
            "* [sad]\n"
            "* [excited]\n"
            "* [angry]\n"
            "* [whisper]\n"
            "* [annoyed]\n"
            "* [appalled]\n"
            "* [thoughtful]\n"
            "* [surprised]\n"
            "* (and similar emotional/delivery directions)\n\n"
            "Non-verbal:\n"
            "* [laughing]\n"
            "* [chuckles]\n"
            "* [sighs]\n"
            "* [clears throat]\n"
            "* [short pause]\n"
            "* [long pause]\n"
            "* [exhales sharply]\n"
            "* [inhales deeply]\n"
            "* (and similar non-verbal sounds)\n\n"
            "## 6. Examples of Enhancement\n\n"
            "Input:\n"
            "\"Are you serious? I can't believe you did that!\"\n\n"
            "Enhanced Output:\n"
            "\"[appalled] Are you serious? [sighs] I can't believe you did that!\"\n\n"
            "---\n\n"
            "Input:\n"
            "\"That's amazing, I didn't know you could sing!\"\n\n"
            "Enhanced Output:\n"
            "\"[laughing] That's amazing, [singing] I didn't know you could sing!\"\n\n"
            "---\n\n"
            "Input:\n"
            "\"I guess you're right. It's just... difficult.\"\n\n"
            "Enhanced Output:\n"
            "\"I guess you're right. [sighs] It's just... [muttering] difficult.\"\n\n"
            "# Instructions Summary\n\n"
            "1. Add audio tags from the audio tags list. These must describe something auditory but only for the voice.\n"
            "2. Enhance emphasis without altering meaning or text.\n"
            "3. Reply ONLY with the enhanced text.\n\n"
            "IMPORTANTE: Para integração com o pipeline, responda como JSON `NarrativePlan` contendo o campo `narration_text` com o diálogo enriquecido."
        )

        user = (
            "Texto original (pt-BR) para receber tags do ElevenLabs:\n"
            f"{clean_text}\n\n"
            "Aplique as tags seguindo TODAS as instruções acima (sem alterar nenhuma palavra original)."
        )

        messages: List[MessageParam] = self._messages(system, user)
        plan = self._completion_typed(
            response_model=NarrativePlan,
            messages=messages,
            enable_reasoning=True,
        )

        return plan.narration_text.strip()

    def generate_narrative(self, topic: str) -> NarrativePlan:
        base_text = self._generate_base_narrative_text(topic)
        tagged_text = self._apply_elevenlabs_audio_tags(base_text)
        return NarrativePlan(narration_text=tagged_text)

    def generate_recipe(
        self,
        topic: str,
        narration_text: str,
        segments: List[dict],
        *,
        audio_path: str,
        audio_duration: float,
        chunk_size: int | None = None,
        video_profile: VideoProfile | None = None,
    ) -> VideoRecipe:
        if not segments:
            raise ValueError("segments list cannot be empty for recipe generation.")

        total_duration = float(
            audio_duration
            or max((float(seg.get("end", 0.0)) for seg in segments), default=0.0)
        )
        audio_path = audio_path or "/audio/narration.mp3"

        metadata = self._generate_recipe_metadata(
            topic=topic,
            narration_text=narration_text,
            total_duration=total_duration,
            audio_path=audio_path,
            video_profile=video_profile,
        )

        try:
            music_plan = self.generate_music_prompt(topic, narration_text, total_duration)
            metadata.background_music_prompt = music_plan.prompt
        except Exception:
            pass

        print(f"[recipe] Phase 1: Analyzing {len(segments)} segments for viral rhythm grouping...")
        try:
            grouping_plan = self.generate_segment_grouping_plan(
                segments=segments,
                topic=topic,
            )
            print(f"[recipe] Viral grouping: {len(segments)} segments → {len(grouping_plan.groups)} scenes")

            grouped_segments = []
            for group in grouping_plan.groups:
                group_segments = [seg for seg in segments if seg.get("index") in group.segment_indices]
                if not group_segments:
                    continue

                group_segments.sort(key=lambda s: s.get("index", 0))

                merged = {
                    "index": group_segments[0].get("index"),
                    "start": group_segments[0].get("start"),
                    "end": group_segments[-1].get("end"),
                    "text": " ".join(s.get("text", "") for s in group_segments).strip(),
                    "_grouped_indices": group.segment_indices,
                    "_reasoning": group.reasoning,
                }
                grouped_segments.append(merged)

            segments_to_process = grouped_segments
            print(f"[recipe] Rhythm-optimized segments ready: {len(segments_to_process)} viral scene groups")
        except Exception as exc:
            print(f"[recipe] Warning: Grouping phase failed ({exc}), using 1:1 mapping")
            segments_to_process = segments

        chunk_size = chunk_size or int(os.getenv("RECIPE_SEGMENT_CHUNK_SIZE", "3"))
        chunk_size = max(1, chunk_size)
        segment_chunks = self._chunk_segments(segments_to_process, chunk_size)

        scenes: List[Scene] = []
        total_scenes = len(segments_to_process)

        for chunk_index, chunk in enumerate(segment_chunks, start=1):
            narration_chunk = " ".join(str(seg.get("text", "")) for seg in chunk)
            chunk_result = self._generate_recipe_scene_chunk(
                topic=topic,
                narration_chunk=narration_chunk,
                segments_chunk=chunk,
                scene_index_start=len(scenes),
                total_scenes=total_scenes,
                video_profile=video_profile,
            )

            if len(chunk_result.scenes) != len(chunk):
                print(
                    f"[recipe] Warning: LLM returned {len(chunk_result.scenes)} scenes "
                    f"for chunk of {len(chunk)} segments (chunk {chunk_index}). Accepting anyway."
                )

            scenes.extend(chunk_result.scenes)

        for idx, scene in enumerate(scenes):
            scene.index = idx

        return VideoRecipe(
            title=metadata.title,
            description=metadata.description,
            tags=metadata.tags,
            language=metadata.language,
            audio=metadata.audio,
            policy=metadata.policy,
            scenes=scenes,
        )


__all__ = [
    "create_instructor_client",
    "TypedLLMClient",
]
