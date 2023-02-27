<h2>COCA - A simple baseline for multimodal knowledge graph</h2>
step 1: !python process_datasets.py<br/>
step 2: !python learn.py --dataset WN9IMG --model COCA --rank 500 --optimizer Adagrad \
        --learning_rate 1e-1 --batch_size 2000 --regularizer N3 --reg 5e-3 --max_epochs 100 \
        --valid 5 -train -id 0 -save -weight
