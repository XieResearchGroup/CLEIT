# CLRN
nvidia-docker run --name clrn_run -it --rm -v /raid/home/dihe/CLRN:/CLRN clrn:publish_1.0  
ctrl+P+Q  
docker exec -it clrn_run /bin/bash

## Data Source
depmap: https://depmap.org/portal/download/  
xena: http://xena.ucsc.edu/public/  
gdsc: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html

## Preprocessing
![preprocessing](./figs/preprocessing.png?raw=true)
![dat_summary](./figs/dat_summary.jpg?raw=true)


## CLRN framework
![clrn_framework](./figs/CLRN_Framework.png?raw=true)

## Gex pre-training and fine-tuning
![gex_train](./figs/gex_train.png?raw=true)

## Mut pre-training and fine-tuning
![mut_train](./figs/mut_train.png?raw=true)


## WGAN transmission
![wgan_trans](./figs/wgan_trans.png?raw=true)

## Performance (0428)
![perf](./figs/performance_0428.png?raw=true)

