set -eu
install -m 600 /mnt/d/chorme/openai-agent-key.pem /home/mapples/.ssh/bohn-aws-deploy.pem
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i /home/mapples/.ssh/bohn-aws-deploy.pem"
cd /home/mapples/projects/mobile-robot-mppi-study
mkdir -p /mnt/d/Projects/mobile-robot-mppi-study/docs/bohn2021_takeover/wsl
for item in status branch log remote; do
 case "$item" in
 status) git status --porcelain=v1;;
 branch) git branch -avv;;
 log) git log -20 --format='%H %aI %s';;
 remote) git remote -v;;
 esac > /mnt/d/Projects/mobile-robot-mppi-study/docs/bohn2021_takeover/wsl/$item.txt
done
git diff --binary > /mnt/d/Projects/mobile-robot-mppi-study/docs/bohn2021_takeover/wsl/dirty.patch
$SSH ubuntu@18.236.70.13 'mkdir -p /data/openai-agent/mobile-robot-mppi-study/experiments /data/openai-agent/mobile-robot-mppi-study/research_artifacts /data/openai-agent/runtime'
rsync -az --partial -e "$SSH" --exclude __pycache__ experiments/bohn2021_reproduction ubuntu@18.236.70.13:/data/openai-agent/mobile-robot-mppi-study/experiments/
rsync -az --partial -e "$SSH" --exclude __pycache__ research_artifacts/bohn2021_reproduction_2026-09-17 ubuntu@18.236.70.13:/data/openai-agent/mobile-robot-mppi-study/research_artifacts/
rsync -az --partial -e "$SSH" /home/mapples/.local/share/bohn2021-python37/ ubuntu@18.236.70.13:/data/openai-agent/runtime/bohn2021-python37/
echo SYNC_COMPLETE
