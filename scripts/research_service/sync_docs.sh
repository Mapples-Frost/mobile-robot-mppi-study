set -eu
cd /home/mapples/projects/mobile-robot-mppi-study
rsync -az -e 'ssh -o BatchMode=yes -i /home/mapples/.ssh/bohn-aws-deploy.pem' docs ubuntu@18.236.70.13:/data/openai-agent/mobile-robot-mppi-study/
