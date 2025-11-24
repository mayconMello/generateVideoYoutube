FROM node:20-bookworm-slim

# Sistema: ffmpeg + toolchain + libs para headless-gl e node-canvas (JPEG/PNG/GIF/Cairo/Pango)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg xvfb xauth ca-certificates \
    python3 python-is-python3 make g++ pkg-config build-essential \
    libxi-dev libglu1-mesa-dev libglew-dev libx11-dev \
    libxext6 libxrender1 libosmesa6 libgl1-mesa-dev libglib2.0-0 \
    libcairo2 libcairo2-dev libpango1.0-0 libpango1.0-dev \
    libpng16-16 libpng-dev libjpeg62-turbo libjpeg62-turbo-dev \
    libgif7 libgif-dev librsvg2-2 librsvg2-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*


# Força renderização por software e garante 'python' no PATH
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
ENV PYTHON=/usr/bin/python
ENV npm_config_python=/usr/bin/python
ENV PORT=3000

# App
WORKDIR /app
COPY server.js .

# Instala o editly localmente (para o require funcionar)
# headless-gl é necessário para transições GL funcionarem sem GPU física
RUN npm init -y && npm install editly gl-transitions gl@^6.0.2

EXPOSE 3000

# Usa xvfb-run para fornecer display virtual necessário para transições GL
# --auto-servernum: escolhe display automaticamente
# --server-args: configura screen virtual 1280x720 com 24-bit depth
CMD ["xvfb-run", "--auto-servernum", "--server-args=-screen 0 1280x720x24", "node", "server.js"]
