import onnx
from onnx.tools import update_model_dims
import numpy as np
import onnx.helper as oh

def modify_onnx(input_onnx, output_onnx, full_vocab_path, cand_vocab_path, batch_size=1, input_length=32, output_length=94):
    
    # load full vocab
    with open(full_vocab_path, 'r', encoding='utf-8') as f:
        full_vocab = f.read().splitlines()
    full_vocab = ['blank'] + full_vocab + [" "]
    full_dict = {c: i for i, c in enumerate(full_vocab)}

    # load candidate vocab get get candidate char index
    with open(cand_vocab_path, 'r', encoding='utf-8') as f:
        cand_vocab = f.read().splitlines()
    cand_vocab = ['blank'] + cand_vocab + [" "]
    cnad_indices = [full_dict[c] for c in cand_vocab]

    model = onnx.load(input_onnx)
    g = model.graph

    # find old nodes location
    first_node_index, last_node_index = None, None
    fc_matmul_index, fc_add_index = None, None
    fc_weight_index, fc_bias_index = None, None
    for i, node in enumerate(g.node):
        if node.name == 'MatMul_0':
            fc_matmul_index = i
        if node.name == 'Add_43':
            fc_add_index = i
        if node.name == "" and node.op_type == "Constant":
            value = onnx.numpy_helper.to_array(node.attribute[0].t)
            if value.shape == (96, len(full_vocab)):
                fc_weight_index = i
                print("full weight shape:", value.shape)
            if value.shape == (len(full_vocab), ):
                fc_bias_index = i
                print("full bias shape:", value.shape)
        if len(node.input) > 0 and node.input[0] == g.input[0].name:
            first_node_index = i
        if len(node.output) > 0 and node.output[0] == g.output[0].name:
            last_node_index = i

    # get candidate weight and bias
    full_weight = onnx.numpy_helper.to_array(g.node[fc_weight_index].attribute[0].t)
    full_bias = onnx.numpy_helper.to_array(g.node[fc_bias_index].attribute[0].t)
    assert full_weight.shape[-1] == full_bias.shape[-1] == len(full_vocab), \
        f"{full_weight.shape[-1]} != {full_bias.shape[-1]} != {len(full_vocab)}"
    cand_weight = full_weight[..., cnad_indices]
    cand_bias = full_bias[cnad_indices]

    # create new constant nodes
    cand_weight_value = oh.make_tensor(
        name='ctc_fc_w_attr_for_plate', data_type=1, dims=cand_weight.shape, 
        vals=cand_weight.flatten())
    cand_weight_node = oh.make_node(
        op_type='Constant', inputs=[], outputs=['ctc_fc_w_attr_for_plate'], 
        name='ctc_fc_w_attr_for_plate', value=cand_weight_value)
    cand_bias_value = oh.make_tensor(
        name='ctc_fc_b_attr_for_plate', data_type=1, dims=cand_bias.shape, 
        vals=cand_bias.flatten())
    cand_bias_node = oh.make_node(
        op_type='Constant', inputs=[], outputs=['ctc_fc_b_attr_for_plate'], 
        name='ctc_fc_b_attr_for_plate', value=cand_bias_value)

    # remove old nodes, insert new nodes and reconnect the edges
    g.node.pop(fc_weight_index)
    g.node.insert(fc_weight_index, cand_weight_node)
    g.node[fc_matmul_index].input[1] = cand_weight_node.output[0]
    g.node.pop(fc_bias_index)
    g.node.insert(fc_bias_index, cand_bias_node)
    g.node[fc_add_index].input[1] = cand_bias_node.output[0]

    # modify input/onnx name
    g.input[0].name = 'input'
    g.output[0].name = 'output'
    g.node[first_node_index].input[0] = g.input[0].name
    g.node[last_node_index].output[0] = g.output[0].name

    # fix input/output shape
    g.input[0].type.tensor_type.shape.dim[0].dim_value = batch_size
    g.input[0].type.tensor_type.shape.dim[1].dim_value = 3
    g.input[0].type.tensor_type.shape.dim[2].dim_value = 32
    g.input[0].type.tensor_type.shape.dim[3].dim_value = input_length
    g.output[0].type.tensor_type.shape.dim[0].dim_value = batch_size
    g.output[0].type.tensor_type.shape.dim[1].dim_value = output_length
    g.output[0].type.tensor_type.shape.dim[2].dim_value = len(cand_vocab)
    # model = update_model_dims.update_inputs_outputs_dims(model, 
    #     {g.input[0].name: [1, 3, 32, input_length]}, 
    #     {g.output[0].name: [1, output_length, len(cand_vocab)]}
    # )
    
    # inference shape
    model = onnx.shape_inference.infer_shapes(model)
    
    onnx.save(model, output_onnx)


def main():
    input_onnx = "ch_ppocr_mobile_v2.0_rec_infer.onnx"
    output_onnx = "plate_recog_bs2_112x32.onnx"
    full_vocab_path = "vocab/ppocr_keys_v1.txt"
    cand_vocab_path = "vocab/city_code_chars.txt"
    modify_onnx(input_onnx, output_onnx, full_vocab_path, cand_vocab_path, 
    	        batch_size=2, input_length=112, output_length=28)

    # input_onnx = "en_number_mobile_v2.0_rec_infer.onnx"
    # output_onnx = "en_number_mobile_v2.0_rec_infer_slim.onnx"
    # full_vocab_path = "vocab/en_dict.txt"
    # cand_vocab_path = "vocab/code_chars.txt"
    # modify_onnx(input_onnx, output_onnx, full_vocab_path, cand_vocab_path, 
    #             batch_size=1, input_length=111, output_length=28)


if __name__ == "__main__":
    main()
