from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import requests
import runpod
import websocket

# Tenta importar utilitários de log do projeto, senão usa print padrão
try:
    from src.media.generation.utils import log_generation, safe_json
except ImportError:
    def log_generation(msg):
        print(f"[RunPod] {msg}")


    def safe_json(obj):
        return json.dumps(obj, default=str)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no"}


class RunPodFluxClient:
    """
    Self-hosted image generation client for Flux.1 Schnell running on a RunPod pod.
    Life cycle: connect to an existing pod (defined in .env) or spin up a new one.
    """

    # --- CONFIGURAÇÃO CRÍTICA ---
    comfy_port = 8188  # Porta padrão do ComfyUI oficial
    gpu_type = "NVIDIA A40"
    image_name = "runpod/stable-diffusion:comfy-ui"

    _instance: Optional["RunPodFluxClient"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "RunPodFluxClient":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        print(os.getenv("RUNPOD_API_KEY"))
        self.api_key = os.getenv("RUNPOD_API_KEY") or ""
        self.volume_id = os.getenv("RUNPOD_VOLUME_ID") or ""
        # ID fixo do pod definido manualmente no .env
        self.forced_pod_id = os.getenv("RUNPOD_POD_ID")

        self.mock_mode = _bool_env("RUNPOD_MOCK_MODE", False)
        self.pod_name = os.getenv("RUNPOD_POD_NAME", "flux-schnell-comfy")

        # Caminho para o arquivo JSON do workflow na mesma pasta deste script
        self._workflow_path = Path(__file__).with_name("workflow_flux.json")

        self._pod_id: Optional[str] = None
        self._base_url: Optional[str] = None
        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self._lock = threading.Lock()

        # Configura a chave para a biblioteca oficial também
        runpod.api_key = self.api_key

    # ------------------------------------------------------------------ lifecycle

    def ensure_pod_active(self) -> str:
        """
        Garante que o pod está rodando e o ComfyUI respondendo.
        Retorna a URL base HTTP do serviço.
        """
        if self.mock_mode:
            self._base_url = "http://127.0.0.1:8188"
            return self._base_url

        if not self.api_key:
            raise RuntimeError("RUNPOD_API_KEY missing; cannot start Flux pod.")

        with self._lock:
            # 1. Se já temos URL validada nesta sessão, retorna
            if self._pod_id and self._base_url:
                return self._base_url

            # 2. Se temos um ID forçado no .env (PRIORIDADE)
            if self.forced_pod_id:
                log_generation(f"Usando Pod ID fixo do .env: {self.forced_pod_id}")
                self._pod_id = self.forced_pod_id
                self._base_url = self._proxy_url(self._pod_id)

                # Verifica se está respondendo
                if not self._is_comfy_ready(self._base_url):
                    # Tenta acordar o pod se estiver desligado
                    try:
                        log_generation(f"Pod {self._pod_id} não responde, tentando Resume...")
                        runpod.resume_pod(self._pod_id)
                        # Pequena pausa para dar tempo do status mudar
                        time.sleep(5)
                    except Exception as e:
                        log_generation(f"Aviso ao tentar resume: {e}")
            else:
                # 3. Lógica automática (busca existente ou cria novo)
                # Útil se você quiser escalar automaticamente no futuro
                pod = self._find_existing_pod()
                if pod and pod.get("id"):
                    self._pod_id = pod["id"]
                    self._base_url = self._proxy_url(self._pod_id)
                else:
                    if not self.volume_id:
                        raise RuntimeError("RUNPOD_VOLUME_ID missing; cannot create NEW pod automatically.")
                    log_generation("Criando novo Pod automático...")
                    created = self._create_pod()
                    pod_id = created.get("id") or created.get("podId")
                    if not pod_id:
                        raise RuntimeError(f"RunPod did not return a pod id: {safe_json(created)}")
                    self._pod_id = pod_id
                    self._base_url = self._proxy_url(self._pod_id)

        # Aguarda o serviço HTTP responder na porta 8188
        log_generation(f"Aguardando ComfyUI em {self._base_url}...")
        self._wait_for_comfy_ready(self._base_url)
        log_generation("ComfyUI Conectado e Pronto!")
        return self._base_url

    def terminate_pod(self) -> None:
        """
        Desliga ou para o Pod para economizar dinheiro.
        """
        if self.mock_mode or not self._pod_id:
            return

        # Se for o pod fixo do .env, apenas paramos (STOP) em vez de destruir (TERMINATE)
        # para não perder a configuração manual e o container
        if self.forced_pod_id and self._pod_id == self.forced_pod_id:
            log_generation(f"Parando (STOP) Pod fixo {self._pod_id}...")
            try:
                runpod.stop_pod(self._pod_id)
            except Exception as e:
                log_generation(f"Erro ao parar pod: {e}")
            return

        # Se for pod criado automaticamente pelo script, destruímos para não gastar storage
        log_generation(f"Destruindo (TERMINATE) Pod automático {self._pod_id}...")
        try:
            try:
                self._terminate_v2(self._pod_id)
            except Exception:
                self._terminate_graphql(self._pod_id)
        except Exception as exc:
            log_generation(f"RunPod terminate failed for pod={self._pod_id}: {exc!r}")
        finally:
            self._pod_id = None
            self._base_url = None

    # ------------------------------------------------------------------ generation

    def generate_image(self, prompt: str, *, width: int = 1080, height: int = 1920) -> Tuple[Optional[str], dict]:
        """
        Gera uma imagem usando o workflow do ComfyUI.
        Retorna (image_url, debug_info).
        """
        debug: Dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "pod_id": self._pod_id,
        }

        try:
            base_url = self.ensure_pod_active()
            debug["base_url"] = base_url

            # Carrega e preenche o workflow
            workflow = self._load_workflow_template()

            # Ajusta nós do workflow (IDs baseados no seu JSON)
            # Seed aleatória
            workflow["3"]["inputs"]["seed"] = random.randint(1, 999999999999)
            # Dimensões
            workflow["5"]["inputs"]["width"] = int(width)
            workflow["5"]["inputs"]["height"] = int(height)
            # Prompt Positivo
            workflow["6"]["inputs"]["text"] = prompt
            # Prompt Negativo Padrão
            neg_prompt = os.getenv("RUNPOD_NEGATIVE_PROMPT", "text, watermark, blurry, low quality, ugly, deformed")
            workflow["7"]["inputs"]["text"] = neg_prompt

            client_id = str(uuid.uuid4())
            ws_url = self._ws_url(base_url, client_id)

            # 1. Conectar WebSocket
            ws = websocket.create_connection(ws_url, timeout=180)

            # 2. Enviar Payload via HTTP
            resp = self._session.post(
                f"{base_url}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=30,
            )
            resp.raise_for_status()
            prompt_id = resp.json().get("prompt_id")
            debug["prompt_id"] = prompt_id

            # 3. Aguardar Resultado via WebSocket
            image_url = self._wait_for_image(ws, base_url, prompt_id, debug)
            ws.close()

            if image_url:
                debug["resolved_url"] = image_url
                return image_url, debug

            debug["error"] = "no_image_returned"
            return None, debug

        except Exception as e:
            debug["error"] = str(e)
            log_generation(f"Flux Error: {e}")
            return None, debug

    # ------------------------------------------------------------------ helpers

    def _load_workflow_template(self) -> dict:
        if not self._workflow_path.exists():
            raise FileNotFoundError(f"Workflow template missing: {self._workflow_path}")
        with open(self._workflow_path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def _ws_url(self, base_url: str, client_id: str) -> str:
        scheme = "wss" if base_url.startswith("https") else "ws"
        return f"{scheme}://{base_url.split('://', 1)[1]}/ws?clientId={client_id}"

    def _wait_for_image(self, ws, base_url, prompt_id, debug) -> Optional[str]:
        start = time.time()
        # Timeout de 5 minutos para geração
        while time.time() - start < 300:
            try:
                out = ws.recv()
                if isinstance(out, str):
                    msg = json.loads(out)
                    # Verifica se a execução terminou
                    if msg['type'] == 'executing':
                        data = msg['data']
                        # node=None significa que o workflow acabou
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            break
                    if msg['type'] == 'execution_error':
                        debug['execution_error'] = msg['data']
                        return None
            except Exception as e:
                debug['ws_error'] = str(e)
                break

        # Busca o nome do arquivo no histórico
        try:
            history_resp = self._session.get(f"{base_url}/history/{prompt_id}", timeout=10)
            history_resp.raise_for_status()
            history = history_resp.json()

            outputs = history[prompt_id]['outputs']

            # Procura o nó de saída (ID 9 no workflow padrão Flux JSON)
            # Se você mudar o workflow, verifique o ID do nó SaveImage
            save_node_id = '9'
            if save_node_id not in outputs:
                # Tenta achar dinamicamente qualquer nó que tenha imagens
                for node_id in outputs:
                    if 'images' in outputs[node_id]:
                        save_node_id = node_id
                        break

            if save_node_id in outputs:
                images_info = outputs[save_node_id]['images']
                if images_info:
                    img_data = images_info[0]
                    filename = img_data['filename']
                    subfolder = img_data['subfolder']
                    folder_type = img_data['type']

                    params = urlencode({
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": folder_type
                    })
                    return f"{base_url}/view?{params}"

        except Exception as e:
            debug["history_error"] = str(e)

        return None

    def _proxy_url(self, pod_id: str) -> str:
        # URL padrão do RunPod proxy
        return f"https://{pod_id}-{self.comfy_port}.proxy.runpod.net"

    def _wait_for_comfy_ready(self, base_url: str) -> None:
        # Tenta conectar por até 10 minutos (boot pode ser lento se baixar modelos)
        start = time.time()
        while time.time() - start < 600:
            if self._is_comfy_ready(base_url):
                return
            time.sleep(5)
        raise RuntimeError(f"ComfyUI não respondeu na porta {self.comfy_port} após 10 minutos.")

    def _is_comfy_ready(self, base_url: str) -> bool:
        try:
            # system_stats é um endpoint leve do ComfyUI
            resp = self._session.get(f"{base_url}/system_stats", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # --- Métodos de fallback para criação automática (se não usar ID fixo) ---

    def _find_existing_pod(self) -> Optional[dict]:
        try:
            pods = self._list_pods_v2()
            for pod in pods:
                # Procura por pods com o nome definido e que não estejam sendo deletados
                name = pod.get("name") or ""
                status = (pod.get("desiredStatus") or pod.get("status") or "").upper()
                if self.pod_name in name and status not in {"TERMINATED", "DELETED"}:
                    return pod
        except Exception as exc:
            log_generation(f"Erro ao listar pods: {exc!r}")
        return None

    def _list_pods_v2(self) -> list:
        resp = self._session.get("https://api.runpod.ai/v2/pods", timeout=30)
        if resp.status_code == 401:
            raise RuntimeError("RunPod API Key inválida.")
        data = resp.json()
        return data.get("data") or data.get("pods") or []

    def _create_pod(self) -> dict:
        # Criação via API v2
        payload = {
            "name": self.pod_name,
            "imageName": self.image_name,
            "gpuTypeId": self.gpu_type,
            "cloudType": "ALL",
            "volumeMountPath": "/workspace",
            "volumeId": self.volume_id,
            "ports": [f"{self.comfy_port}/http"],
            "containerDiskInGb": 10,
            "minVcpuCount": 4,
            "minMemoryInGb": 24,
            "dockerArgs": f"python main.py --listen --port {self.comfy_port}"  # Tenta forçar start
        }
        resp = self._session.post("https://api.runpod.ai/v2/pods", json=payload, timeout=40)
        resp.raise_for_status()
        return resp.json().get("data") or resp.json()

    def _terminate_v2(self, pod_id: str) -> None:
        self._session.delete(f"https://api.runpod.ai/v2/pods/{pod_id}", timeout=30)

    def _terminate_graphql(self, pod_id: str) -> None:
        # Fallback se v2 falhar
        pass


# Instância global para uso no projeto
flux_client = RunPodFluxClient.instance()