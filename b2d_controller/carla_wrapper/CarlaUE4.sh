#!/bin/sh
# Thin wrapper forcing -opengl: Vulkan (this build's default RHI) segfaults on
# world (re)load in -RenderOffScreen headless mode on this GPU/driver combo
# (confirmed via isolated client.load_world() repro); -opengl does not.
exec /home/ailab/carla/CARLA_0.9.15/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -opengl \
  r.Streaming.PoolSize=512 r.Streaming.PoolSizeMemoryBudget=512 r.TextureStreaming=0 \
  "$@"
