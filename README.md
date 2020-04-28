# CLRN
nvidia-docker run --name clrn_run -it --rm -v /raid/home/dihe/CLRN:/CLRN clrn:publish_1.0
ctrl+P+Q
docker exec -it clrn:publish_1.0 /bin/bash
