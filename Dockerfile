FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Base build & system deps + VTK + OpenCV
RUN apt-get update && apt-get install -y \
  build-essential \
  pkg-config \
  libhdf5-dev \
  liblz4-dev \
  cmake \
  git \
  wget \
  unzip \
  # LASlib deps
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  zlib1g-dev \
  libproj-dev \
  liblzma-dev \
  libjbig-dev \
  libzstd-dev \
  libgeotiff-dev \
  libwebp-dev \
  libsqlite3-dev \
  # Linear algebra / PCL deps
  libsuperlu-dev \
  libpcl-dev \
  # Optional: PAPI deps (still built from source)
  python3 \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/octrees-benchmark

# Copy project
COPY . .

RUN mkdir -p lib

# Build LASlib
RUN chmod +x scripts/install_laslib.sh && ./scripts/install_laslib.sh

# Build Picotree
RUN chmod +x scripts/install_picotree.sh && ./scripts/install_picotree.sh

# Build PAPI
RUN chmod +x scripts/install_papi.sh && ./scripts/install_papi.sh

# Configure and build
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release . && \
  cmake --build build -j"$(nproc)"

CMD ["./build/octrees-benchmark", "--help"]