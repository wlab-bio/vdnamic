# non-hierarchical GSE
python ../../main.py GSE -path ./ -inference_dim 3 -inference_eignum 25 -final_eignum 250
# hierarchical GSE
# python ../../main.py GSE -path ./ -inference_dim 3 -inference_eignum 25 -final_eignum 250 -sub_num 25 -sub_size 5000 -filter_criterion 0.0
# hierarchical GSE with 0.2% filter
# python ../../main.py GSE -path ./ -inference_dim 3 -inference_eignum 25 -final_eignum 250 -sub_num 25 -sub_size 5000 -filter_criterion 0.1,0.1
