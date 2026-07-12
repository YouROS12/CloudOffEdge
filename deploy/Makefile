# Paper 2 deploy ops shortcuts.
#
# Quickstart (from repo root):
#   make -C paper2/deploy help
#
# Common workflow:
#   make -C paper2/deploy package                     # bundle ckpts -> checkpoints.tar.gz
#   make -C paper2/deploy deploy-edge   PI=user@pi    # rsync + run setup.sh on the Pi
#   make -C paper2/deploy deploy-cloud  JETSON=user@jetson
#   make -C paper2/deploy logs-edge     PI=user@pi
#   make -C paper2/deploy test-local                  # spin up both servers locally on different ports
#
# Variables (override with `make foo VAR=value`):
#   PI               ssh target for the edge device, e.g. youssef@my-pi
#   JETSON           ssh target for the cloud device
#   CKPT_URL         public URL for checkpoints.tar.gz (passed as CHECKPOINTS_URL)
#   IMAGE_DIR        local folder of test images for `make batch`
#   EDGE_URL         http://<pi-tailscale-name>:8000

PY              ?= python
PIP             ?= $(PY) -m pip
PI              ?= pi@raspberrypi.local
JETSON          ?= nvidia@jetson.local
CKPT_URL        ?=
IMAGE_DIR       ?=
EDGE_URL        ?= http://localhost:8000

# Repo paths (Makefile lives at paper2/deploy/Makefile)
DEPLOY_DIR      := $(CURDIR)
PAPER2_DIR      := $(abspath $(DEPLOY_DIR)/..)
TARBALL         := $(DEPLOY_DIR)/checkpoints.tar.gz
EDGE_CKPT       ?= $(PAPER2_DIR)/outputs/edge_model_v3l/best.pt
CLOUD_CKPT      ?= $(PAPER2_DIR)/outputs/cloud_head/qualitygate_fusion_3L.pt

# Remote install dir on each device
REMOTE_DIR      := /opt/plant-deploy

# Files to ship via rsync (everything under paper2/deploy + paper1/ for class names)
RSYNC_EXCLUDES  := --exclude '__pycache__' --exclude '*.pyc' \
                   --exclude '.venv' --exclude 'checkpoints.tar.gz' \
                   --exclude 'logs' --exclude '.git'

.PHONY: help
help:
	@echo "Paper 2 deploy targets:"
	@echo "  package          Build $(TARBALL) from local outputs/"
	@echo "  deploy-edge      rsync repo + run setup.sh edge   (PI=$(PI))"
	@echo "  deploy-cloud     rsync repo + run setup.sh cloud  (JETSON=$(JETSON))"
	@echo "  ship-ckpts-edge  scp $(TARBALL) to Pi  + extract under checkpoints/"
	@echo "  ship-ckpts-cloud scp $(TARBALL) to Jet + extract under checkpoints/"
	@echo "  start-edge / start-cloud      systemctl restart on the device"
	@echo "  stop-edge  / stop-cloud       systemctl stop on the device"
	@echo "  status-edge / status-cloud    systemctl status on the device"
	@echo "  logs-edge / logs-cloud        journalctl -fu plant-{edge,cloud}"
	@echo "  health-edge / health-cloud    curl /health (uses EDGE_URL or JETSON)"
	@echo "  batch            Run batch_predict.py against EDGE_URL"
	@echo "                    (set IMAGE_DIR=/path/to/test/images)"
	@echo "  test-local       Run cloud (8001) + edge (8000) locally for smoke test"
	@echo "  clean            Remove $(TARBALL) and __pycache__"
	@echo ""
	@echo "Dashboard (run on your laptop):"
	@echo "  dashboard         Run dashboard server on :9000 (Ctrl-C to stop)"
	@echo "  dashboard-deps    pip install dashboard requirements"
	@echo "  open-dashboard    Open http://localhost:9000 in your browser"

# ---------------------------------------------------------------------------
# Local: package + sanity
# ---------------------------------------------------------------------------

.PHONY: package
package:
	cd $(PAPER2_DIR) && $(PY) deploy/package_checkpoints.py \
	  --edge-ckpt $(EDGE_CKPT) \
	  --cloud-ckpt $(CLOUD_CKPT) \
	  --out $(TARBALL)
	@echo "[ok] tarball: $(TARBALL)"
	@ls -lh $(TARBALL) | awk '{print "      "$$5"  "$$9}'

.PHONY: clean
clean:
	rm -f $(TARBALL)
	find $(DEPLOY_DIR) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "[ok] cleaned"

# ---------------------------------------------------------------------------
# Remote: rsync repo, then run setup.sh on the device
# ---------------------------------------------------------------------------

define _ensure_remote_dir
	ssh $(1) 'sudo mkdir -p $(REMOTE_DIR) && sudo chown $$USER:$$USER $(REMOTE_DIR)'
endef

.PHONY: rsync-edge
rsync-edge:
	$(call _ensure_remote_dir,$(PI))
	rsync -a $(RSYNC_EXCLUDES) $(PAPER2_DIR)/ $(PI):$(REMOTE_DIR)/

.PHONY: rsync-cloud
rsync-cloud:
	$(call _ensure_remote_dir,$(JETSON))
	rsync -a $(RSYNC_EXCLUDES) $(PAPER2_DIR)/ $(JETSON):$(REMOTE_DIR)/

.PHONY: ship-ckpts-edge
ship-ckpts-edge: $(TARBALL)
	scp $(TARBALL) $(PI):/tmp/checkpoints.tar.gz
	ssh $(PI) 'mkdir -p $(REMOTE_DIR)/deploy/checkpoints \
	  && tar -xzf /tmp/checkpoints.tar.gz -C $(REMOTE_DIR)/deploy/checkpoints/ \
	  && rm /tmp/checkpoints.tar.gz \
	  && ls -lh $(REMOTE_DIR)/deploy/checkpoints/'

.PHONY: ship-ckpts-cloud
ship-ckpts-cloud: $(TARBALL)
	scp $(TARBALL) $(JETSON):/tmp/checkpoints.tar.gz
	ssh $(JETSON) 'mkdir -p $(REMOTE_DIR)/deploy/checkpoints \
	  && tar -xzf /tmp/checkpoints.tar.gz -C $(REMOTE_DIR)/deploy/checkpoints/ \
	  && rm /tmp/checkpoints.tar.gz \
	  && ls -lh $(REMOTE_DIR)/deploy/checkpoints/'

$(TARBALL):
	@echo "missing $(TARBALL); run: make package" && exit 1

.PHONY: deploy-edge
deploy-edge: rsync-edge ship-ckpts-edge
	ssh $(PI) 'cd $(REMOTE_DIR) && bash deploy/setup.sh edge'

.PHONY: deploy-cloud
deploy-cloud: rsync-cloud ship-ckpts-cloud
	ssh $(JETSON) 'cd $(REMOTE_DIR) && bash deploy/setup.sh cloud'

# Same as deploy-* but expects CHECKPOINTS_URL instead of pushing a tarball
.PHONY: deploy-edge-url
deploy-edge-url: rsync-edge
	@test -n "$(CKPT_URL)" || (echo "set CKPT_URL=https://..."; exit 2)
	ssh $(PI) 'cd $(REMOTE_DIR) && CHECKPOINTS_URL=$(CKPT_URL) bash deploy/setup.sh edge'

.PHONY: deploy-cloud-url
deploy-cloud-url: rsync-cloud
	@test -n "$(CKPT_URL)" || (echo "set CKPT_URL=https://..."; exit 2)
	ssh $(JETSON) 'cd $(REMOTE_DIR) && CHECKPOINTS_URL=$(CKPT_URL) bash deploy/setup.sh cloud'

# ---------------------------------------------------------------------------
# Service control
# ---------------------------------------------------------------------------

.PHONY: start-edge stop-edge restart-edge status-edge logs-edge
start-edge:    ; ssh $(PI)     'sudo systemctl start  plant-edge'
stop-edge:     ; ssh $(PI)     'sudo systemctl stop   plant-edge'
restart-edge:  ; ssh $(PI)     'sudo systemctl restart plant-edge'
status-edge:   ; ssh $(PI)     'systemctl status plant-edge --no-pager || true'
logs-edge:     ; ssh $(PI)     'journalctl -u plant-edge -f --no-pager'

.PHONY: start-cloud stop-cloud restart-cloud status-cloud logs-cloud
start-cloud:   ; ssh $(JETSON) 'sudo systemctl start  plant-cloud'
stop-cloud:    ; ssh $(JETSON) 'sudo systemctl stop   plant-cloud'
restart-cloud: ; ssh $(JETSON) 'sudo systemctl restart plant-cloud'
status-cloud:  ; ssh $(JETSON) 'systemctl status plant-cloud --no-pager || true'
logs-cloud:    ; ssh $(JETSON) 'journalctl -u plant-cloud -f --no-pager'

# ---------------------------------------------------------------------------
# Healthchecks + batch run
# ---------------------------------------------------------------------------

.PHONY: health-edge
health-edge:
	curl -fsS $(EDGE_URL)/health | $(PY) -m json.tool

.PHONY: health-cloud
health-cloud:
	@test -n "$(CLOUD_URL)" || (echo "set CLOUD_URL=http://<jetson-tailscale>:8001"; exit 2)
	curl -fsS $(CLOUD_URL)/health | $(PY) -m json.tool

.PHONY: batch
batch:
	@test -n "$(IMAGE_DIR)" || (echo "set IMAGE_DIR=/path/to/test/images"; exit 2)
	cd $(PAPER2_DIR) && $(PY) deploy/client/batch_predict.py \
	  --image-dir "$(IMAGE_DIR)" \
	  --edge-url $(EDGE_URL) \
	  --out-csv results/deploy_run_$(shell date +%Y%m%d_%H%M%S).csv \
	  --label-from-dirname

# ---------------------------------------------------------------------------
# Local two-process smoke test (no Tailscale needed)
# ---------------------------------------------------------------------------

.PHONY: test-local
test-local:
	@echo "[smoke] starting cloud on :8001 and edge on :8000"
	@cd $(DEPLOY_DIR) && DEPLOY_ROLE=cloud SERVER_PORT=8001 CLOUD_DEVICE=$${CLOUD_DEVICE:-cpu} CLOUD_SEQUENTIAL=1 \
	  $(PY) -m cloud.server &  echo $$! > /tmp/.plant-cloud.pid
	@sleep 5
	@cd $(DEPLOY_DIR) && DEPLOY_ROLE=edge SERVER_PORT=8000 CLOUD_URL=http://localhost:8001 EDGE_DEVICE=cpu \
	  $(PY) -m edge.server &   echo $$! > /tmp/.plant-edge.pid
	@sleep 5
	@curl -fsS http://localhost:8000/health || true
	@echo
	@echo "[smoke] kill with: kill $$(cat /tmp/.plant-edge.pid /tmp/.plant-cloud.pid)"

.PHONY: test-local-stop
test-local-stop:
	-@kill $$(cat /tmp/.plant-edge.pid 2>/dev/null) 2>/dev/null || true
	-@kill $$(cat /tmp/.plant-cloud.pid 2>/dev/null) 2>/dev/null || true
	-@rm -f /tmp/.plant-edge.pid /tmp/.plant-cloud.pid
	@echo "[smoke] stopped"

# ---------------------------------------------------------------------------
# Dashboard (run on your laptop; nodes POST heartbeats to it)
# ---------------------------------------------------------------------------

DASHBOARD_PORT ?= 9000

.PHONY: dashboard-deps
dashboard-deps:
	$(PIP) install -r $(DEPLOY_DIR)/dashboard/requirements.txt

.PHONY: dashboard
dashboard:
	@echo "[dashboard] serving on http://0.0.0.0:$(DASHBOARD_PORT)"
	@echo "[dashboard] set DASHBOARD_URL=http://<this-laptop-tailscale>:$(DASHBOARD_PORT) on each Pi/Jetson"
	cd $(DEPLOY_DIR) && DASHBOARD_HOST=0.0.0.0 DASHBOARD_PORT=$(DASHBOARD_PORT) \
	  $(PY) -m dashboard.server

.PHONY: open-dashboard
open-dashboard:
	@$(PY) -c "import webbrowser; webbrowser.open('http://localhost:$(DASHBOARD_PORT)')"
