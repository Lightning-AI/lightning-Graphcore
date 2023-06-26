# Creation IPU agent pool

## Connecting to the IPU instance

1. You need to get SSH access granted by Graphcore
1. `ssh jirkab@lr76-4c.usclt-pod1.graphcloud.ai` (set your name obviously)
1. Try to run [monitor](https://www.docker.com/blog/graphcore-poplar-sdk-container-images-now-available-on-docker-hub/):
   ```bash
   docker run --rm \
       --ulimit memlock=-1:-1 \
       --net=host \
       --cap-add=IPC_LOCK \
       --device=/dev/infiniband \
       --ipc=host \
       -v ~/.ipuof.conf.d/:/etc/ipuof.conf.d \
       -it graphcore/tools gc-info -l
   ```

## Starting the Azure runners/pool

1. [Generate your PAT](https://docs.microsoft.com/en-us/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate) with permission to the agent pool
1. Start a `screen`
1. Run the docker agent with your account’s `AZP_TOKEN`
   ```bash
   docker run -t \
       -v /dev:/dev \
       -e AZP_URL="https://dev.azure.com/Lightning-AI" \
       -e AZP_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXXXX" \
       -e AZP_AGENT_NAME="lr76-4c-poplar-1-1" \
       -e AZP_POOL="graphcore-ipus" \
       -v /mnt/public:/mnt/public:ro \
     -v /opt/poplar:/opt/poplar:ro \
     --ulimit memlock=-1:-1 \
       --net=host \
       --cap-add=IPC_LOCK \
       --device=/dev/infiniband \
       --ipc=host \
       -v ~/.ipuof.conf.d/:/etc/ipuof.conf.d \
       pytorchlightning/lightning-graphcore:ipu-ci-runner-py3.8
   ```
