
            import json
            from typing import Dict, Any, Optional
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            from .config import ModelConfig

            IR_JSON = r"""{
  "name": "decoder_only_v1",
  "tensors": {
    "input_ids": {
      "name": "input_ids",
      "shape": [
        -1,
        2048
      ],
      "dtype": "int32",
      "node": null,
      "consumers": [
        "token_embeddings"
      ]
    },
    "token_embeddings_out_1": {
      "name": "token_embeddings_out_1",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "token_embeddings",
      "consumers": [
        "rope_positional"
      ]
    },
    "rope_positional_out_2": {
      "name": "rope_positional_out_2",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "rope_positional",
      "consumers": [
        "input_norm"
      ]
    },
    "input_norm_out_3": {
      "name": "input_norm_out_3",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "input_norm",
      "consumers": [
        "layer_0_attn_norm",
        "layer_0_attn_residual"
      ]
    },
    "layer_0_attn_norm_out_4": {
      "name": "layer_0_attn_norm_out_4",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_0_attn_norm",
      "consumers": [
        "layer_0_q_proj",
        "layer_0_k_proj",
        "layer_0_v_proj"
      ]
    },
    "layer_0_q_proj_out_5": {
      "name": "layer_0_q_proj_out_5",
      "shape": [
        -1,
        -1,
        203
      ],
      "dtype": "float32",
      "node": "layer_0_q_proj",
      "consumers": [
        "layer_0_attn"
      ]
    },
    "layer_0_k_proj_out_6": {
      "name": "layer_0_k_proj_out_6",
      "shape": [
        -1,
        -1,
        29
      ],
      "dtype": "float32",
      "node": "layer_0_k_proj",
      "consumers": [
        "layer_0_attn"
      ]
    },
    "layer_0_v_proj_out_7": {
      "name": "layer_0_v_proj_out_7",
      "shape": [
        -1,
        -1,
        29
      ],
      "dtype": "float32",
      "node": "layer_0_v_proj",
      "consumers": [
        "layer_0_attn"
      ]
    },
    "layer_0_attn_out_8": {
      "name": "layer_0_attn_out_8",
      "shape": [
        -1,
        -1,
        203
      ],
      "dtype": "float32",
      "node": "layer_0_attn",
      "consumers": [
        "layer_0_attn_out_proj"
      ]
    },
    "layer_0_attn_out_proj_out_9": {
      "name": "layer_0_attn_out_proj_out_9",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_0_attn_out_proj",
      "consumers": [
        "layer_0_attn_residual"
      ]
    },
    "layer_0_attn_residual_out_10": {
      "name": "layer_0_attn_residual_out_10",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_0_attn_residual",
      "consumers": [
        "layer_0_mlp_norm",
        "layer_0_mlp_residual"
      ]
    },
    "layer_0_mlp_norm_out_11": {
      "name": "layer_0_mlp_norm_out_11",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_0_mlp_norm",
      "consumers": [
        "layer_0_gate_proj",
        "layer_0_up_proj"
      ]
    },
    "layer_0_gate_proj_out_12": {
      "name": "layer_0_gate_proj_out_12",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_0_gate_proj",
      "consumers": [
        "layer_0_swiglu"
      ]
    },
    "layer_0_up_proj_out_13": {
      "name": "layer_0_up_proj_out_13",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_0_up_proj",
      "consumers": [
        "layer_0_swiglu"
      ]
    },
    "layer_0_swiglu_out_14": {
      "name": "layer_0_swiglu_out_14",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_0_swiglu",
      "consumers": [
        "layer_0_down_proj"
      ]
    },
    "layer_0_down_proj_out_15": {
      "name": "layer_0_down_proj_out_15",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_0_down_proj",
      "consumers": [
        "layer_0_mlp_residual"
      ]
    },
    "layer_0_mlp_residual_out_16": {
      "name": "layer_0_mlp_residual_out_16",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_0_mlp_residual",
      "consumers": [
        "layer_1_attn_norm",
        "layer_1_attn_residual"
      ]
    },
    "layer_1_attn_norm_out_17": {
      "name": "layer_1_attn_norm_out_17",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_1_attn_norm",
      "consumers": [
        "layer_1_q_proj",
        "layer_1_k_proj",
        "layer_1_v_proj"
      ]
    },
    "layer_1_q_proj_out_18": {
      "name": "layer_1_q_proj_out_18",
      "shape": [
        -1,
        -1,
        203
      ],
      "dtype": "float32",
      "node": "layer_1_q_proj",
      "consumers": [
        "layer_1_attn"
      ]
    },
    "layer_1_k_proj_out_19": {
      "name": "layer_1_k_proj_out_19",
      "shape": [
        -1,
        -1,
        29
      ],
      "dtype": "float32",
      "node": "layer_1_k_proj",
      "consumers": [
        "layer_1_attn"
      ]
    },
    "layer_1_v_proj_out_20": {
      "name": "layer_1_v_proj_out_20",
      "shape": [
        -1,
        -1,
        29
      ],
      "dtype": "float32",
      "node": "layer_1_v_proj",
      "consumers": [
        "layer_1_attn"
      ]
    },
    "layer_1_attn_out_21": {
      "name": "layer_1_attn_out_21",
      "shape": [
        -1,
        -1,
        203
      ],
      "dtype": "float32",
      "node": "layer_1_attn",
      "consumers": [
        "layer_1_attn_out_proj"
      ]
    },
    "layer_1_attn_out_proj_out_22": {
      "name": "layer_1_attn_out_proj_out_22",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_1_attn_out_proj",
      "consumers": [
        "layer_1_attn_residual"
      ]
    },
    "layer_1_attn_residual_out_23": {
      "name": "layer_1_attn_residual_out_23",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_1_attn_residual",
      "consumers": [
        "layer_1_mlp_norm",
        "layer_1_mlp_residual"
      ]
    },
    "layer_1_mlp_norm_out_24": {
      "name": "layer_1_mlp_norm_out_24",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_1_mlp_norm",
      "consumers": [
        "layer_1_gate_proj",
        "layer_1_up_proj"
      ]
    },
    "layer_1_gate_proj_out_25": {
      "name": "layer_1_gate_proj_out_25",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_1_gate_proj",
      "consumers": [
        "layer_1_swiglu"
      ]
    },
    "layer_1_up_proj_out_26": {
      "name": "layer_1_up_proj_out_26",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_1_up_proj",
      "consumers": [
        "layer_1_swiglu"
      ]
    },
    "layer_1_swiglu_out_27": {
      "name": "layer_1_swiglu_out_27",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_1_swiglu",
      "consumers": [
        "layer_1_down_proj"
      ]
    },
    "layer_1_down_proj_out_28": {
      "name": "layer_1_down_proj_out_28",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_1_down_proj",
      "consumers": [
        "layer_1_mlp_residual"
      ]
    },
    "layer_1_mlp_residual_out_29": {
      "name": "layer_1_mlp_residual_out_29",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_1_mlp_residual",
      "consumers": [
        "layer_2_attn_norm",
        "layer_2_attn_residual"
      ]
    },
    "layer_2_attn_norm_out_30": {
      "name": "layer_2_attn_norm_out_30",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_2_attn_norm",
      "consumers": [
        "layer_2_q_proj",
        "layer_2_k_proj",
        "layer_2_v_proj"
      ]
    },
    "layer_2_q_proj_out_31": {
      "name": "layer_2_q_proj_out_31",
      "shape": [
        -1,
        -1,
        203
      ],
      "dtype": "float32",
      "node": "layer_2_q_proj",
      "consumers": [
        "layer_2_attn"
      ]
    },
    "layer_2_k_proj_out_32": {
      "name": "layer_2_k_proj_out_32",
      "shape": [
        -1,
        -1,
        29
      ],
      "dtype": "float32",
      "node": "layer_2_k_proj",
      "consumers": [
        "layer_2_attn"
      ]
    },
    "layer_2_v_proj_out_33": {
      "name": "layer_2_v_proj_out_33",
      "shape": [
        -1,
        -1,
        29
      ],
      "dtype": "float32",
      "node": "layer_2_v_proj",
      "consumers": [
        "layer_2_attn"
      ]
    },
    "layer_2_attn_out_34": {
      "name": "layer_2_attn_out_34",
      "shape": [
        -1,
        -1,
        203
      ],
      "dtype": "float32",
      "node": "layer_2_attn",
      "consumers": [
        "layer_2_attn_out_proj"
      ]
    },
    "layer_2_attn_out_proj_out_35": {
      "name": "layer_2_attn_out_proj_out_35",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_2_attn_out_proj",
      "consumers": [
        "layer_2_attn_residual"
      ]
    },
    "layer_2_attn_residual_out_36": {
      "name": "layer_2_attn_residual_out_36",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_2_attn_residual",
      "consumers": [
        "layer_2_mlp_norm",
        "layer_2_mlp_residual"
      ]
    },
    "layer_2_mlp_norm_out_37": {
      "name": "layer_2_mlp_norm_out_37",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_2_mlp_norm",
      "consumers": [
        "layer_2_gate_proj",
        "layer_2_up_proj"
      ]
    },
    "layer_2_gate_proj_out_38": {
      "name": "layer_2_gate_proj_out_38",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_2_gate_proj",
      "consumers": [
        "layer_2_swiglu"
      ]
    },
    "layer_2_up_proj_out_39": {
      "name": "layer_2_up_proj_out_39",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_2_up_proj",
      "consumers": [
        "layer_2_swiglu"
      ]
    },
    "layer_2_swiglu_out_40": {
      "name": "layer_2_swiglu_out_40",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_2_swiglu",
      "consumers": [
        "layer_2_down_proj"
      ]
    },
    "layer_2_down_proj_out_41": {
      "name": "layer_2_down_proj_out_41",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_2_down_proj",
      "consumers": [
        "layer_2_mlp_residual"
      ]
    },
    "layer_2_mlp_residual_out_42": {
      "name": "layer_2_mlp_residual_out_42",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_2_mlp_residual",
      "consumers": [
        "layer_3_attn_norm",
        "layer_3_attn_residual"
      ]
    },
    "layer_3_attn_norm_out_43": {
      "name": "layer_3_attn_norm_out_43",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_3_attn_norm",
      "consumers": [
        "layer_3_q_proj",
        "layer_3_k_proj",
        "layer_3_v_proj"
      ]
    },
    "layer_3_q_proj_out_44": {
      "name": "layer_3_q_proj_out_44",
      "shape": [
        -1,
        -1,
        203
      ],
      "dtype": "float32",
      "node": "layer_3_q_proj",
      "consumers": [
        "layer_3_attn"
      ]
    },
    "layer_3_k_proj_out_45": {
      "name": "layer_3_k_proj_out_45",
      "shape": [
        -1,
        -1,
        29
      ],
      "dtype": "float32",
      "node": "layer_3_k_proj",
      "consumers": [
        "layer_3_attn"
      ]
    },
    "layer_3_v_proj_out_46": {
      "name": "layer_3_v_proj_out_46",
      "shape": [
        -1,
        -1,
        29
      ],
      "dtype": "float32",
      "node": "layer_3_v_proj",
      "consumers": [
        "layer_3_attn"
      ]
    },
    "layer_3_attn_out_47": {
      "name": "layer_3_attn_out_47",
      "shape": [
        -1,
        -1,
        203
      ],
      "dtype": "float32",
      "node": "layer_3_attn",
      "consumers": [
        "layer_3_attn_out_proj"
      ]
    },
    "layer_3_attn_out_proj_out_48": {
      "name": "layer_3_attn_out_proj_out_48",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_3_attn_out_proj",
      "consumers": [
        "layer_3_attn_residual"
      ]
    },
    "layer_3_attn_residual_out_49": {
      "name": "layer_3_attn_residual_out_49",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_3_attn_residual",
      "consumers": [
        "layer_3_mlp_norm",
        "layer_3_mlp_residual"
      ]
    },
    "layer_3_mlp_norm_out_50": {
      "name": "layer_3_mlp_norm_out_50",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_3_mlp_norm",
      "consumers": [
        "layer_3_gate_proj",
        "layer_3_up_proj"
      ]
    },
    "layer_3_gate_proj_out_51": {
      "name": "layer_3_gate_proj_out_51",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_3_gate_proj",
      "consumers": [
        "layer_3_swiglu"
      ]
    },
    "layer_3_up_proj_out_52": {
      "name": "layer_3_up_proj_out_52",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_3_up_proj",
      "consumers": [
        "layer_3_swiglu"
      ]
    },
    "layer_3_swiglu_out_53": {
      "name": "layer_3_swiglu_out_53",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_3_swiglu",
      "consumers": [
        "layer_3_down_proj"
      ]
    },
    "layer_3_down_proj_out_54": {
      "name": "layer_3_down_proj_out_54",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_3_down_proj",
      "consumers": [
        "layer_3_mlp_residual"
      ]
    },
    "layer_3_mlp_residual_out_55": {
      "name": "layer_3_mlp_residual_out_55",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_3_mlp_residual",
      "consumers": [
        "layer_4_attn_norm",
        "layer_4_attn_residual"
      ]
    },
    "layer_4_attn_norm_out_56": {
      "name": "layer_4_attn_norm_out_56",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_4_attn_norm",
      "consumers": [
        "layer_4_q_proj",
        "layer_4_k_proj",
        "layer_4_v_proj"
      ]
    },
    "layer_4_q_proj_out_57": {
      "name": "layer_4_q_proj_out_57",
      "shape": [
        -1,
        -1,
        203
      ],
      "dtype": "float32",
      "node": "layer_4_q_proj",
      "consumers": [
        "layer_4_attn"
      ]
    },
    "layer_4_k_proj_out_58": {
      "name": "layer_4_k_proj_out_58",
      "shape": [
        -1,
        -1,
        29
      ],
      "dtype": "float32",
      "node": "layer_4_k_proj",
      "consumers": [
        "layer_4_attn"
      ]
    },
    "layer_4_v_proj_out_59": {
      "name": "layer_4_v_proj_out_59",
      "shape": [
        -1,
        -1,
        29
      ],
      "dtype": "float32",
      "node": "layer_4_v_proj",
      "consumers": [
        "layer_4_attn"
      ]
    },
    "layer_4_attn_out_60": {
      "name": "layer_4_attn_out_60",
      "shape": [
        -1,
        -1,
        203
      ],
      "dtype": "float32",
      "node": "layer_4_attn",
      "consumers": [
        "layer_4_attn_out_proj"
      ]
    },
    "layer_4_attn_out_proj_out_61": {
      "name": "layer_4_attn_out_proj_out_61",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_4_attn_out_proj",
      "consumers": [
        "layer_4_attn_residual"
      ]
    },
    "layer_4_attn_residual_out_62": {
      "name": "layer_4_attn_residual_out_62",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_4_attn_residual",
      "consumers": [
        "layer_4_mlp_norm",
        "layer_4_mlp_residual"
      ]
    },
    "layer_4_mlp_norm_out_63": {
      "name": "layer_4_mlp_norm_out_63",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_4_mlp_norm",
      "consumers": [
        "layer_4_gate_proj",
        "layer_4_up_proj"
      ]
    },
    "layer_4_gate_proj_out_64": {
      "name": "layer_4_gate_proj_out_64",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_4_gate_proj",
      "consumers": [
        "layer_4_swiglu"
      ]
    },
    "layer_4_up_proj_out_65": {
      "name": "layer_4_up_proj_out_65",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_4_up_proj",
      "consumers": [
        "layer_4_swiglu"
      ]
    },
    "layer_4_swiglu_out_66": {
      "name": "layer_4_swiglu_out_66",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_4_swiglu",
      "consumers": [
        "layer_4_down_proj"
      ]
    },
    "layer_4_down_proj_out_67": {
      "name": "layer_4_down_proj_out_67",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_4_down_proj",
      "consumers": [
        "layer_4_mlp_residual"
      ]
    },
    "layer_4_mlp_residual_out_68": {
      "name": "layer_4_mlp_residual_out_68",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_4_mlp_residual",
      "consumers": [
        "layer_5_attn_norm",
        "layer_5_attn_residual"
      ]
    },
    "layer_5_attn_norm_out_69": {
      "name": "layer_5_attn_norm_out_69",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_5_attn_norm",
      "consumers": [
        "layer_5_q_proj",
        "layer_5_k_proj",
        "layer_5_v_proj"
      ]
    },
    "layer_5_q_proj_out_70": {
      "name": "layer_5_q_proj_out_70",
      "shape": [
        -1,
        -1,
        203
      ],
      "dtype": "float32",
      "node": "layer_5_q_proj",
      "consumers": [
        "layer_5_attn"
      ]
    },
    "layer_5_k_proj_out_71": {
      "name": "layer_5_k_proj_out_71",
      "shape": [
        -1,
        -1,
        29
      ],
      "dtype": "float32",
      "node": "layer_5_k_proj",
      "consumers": [
        "layer_5_attn"
      ]
    },
    "layer_5_v_proj_out_72": {
      "name": "layer_5_v_proj_out_72",
      "shape": [
        -1,
        -1,
        29
      ],
      "dtype": "float32",
      "node": "layer_5_v_proj",
      "consumers": [
        "layer_5_attn"
      ]
    },
    "layer_5_attn_out_73": {
      "name": "layer_5_attn_out_73",
      "shape": [
        -1,
        -1,
        203
      ],
      "dtype": "float32",
      "node": "layer_5_attn",
      "consumers": [
        "layer_5_attn_out_proj"
      ]
    },
    "layer_5_attn_out_proj_out_74": {
      "name": "layer_5_attn_out_proj_out_74",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_5_attn_out_proj",
      "consumers": [
        "layer_5_attn_residual"
      ]
    },
    "layer_5_attn_residual_out_75": {
      "name": "layer_5_attn_residual_out_75",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_5_attn_residual",
      "consumers": [
        "layer_5_mlp_norm",
        "layer_5_mlp_residual"
      ]
    },
    "layer_5_mlp_norm_out_76": {
      "name": "layer_5_mlp_norm_out_76",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_5_mlp_norm",
      "consumers": [
        "layer_5_gate_proj",
        "layer_5_up_proj"
      ]
    },
    "layer_5_gate_proj_out_77": {
      "name": "layer_5_gate_proj_out_77",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_5_gate_proj",
      "consumers": [
        "layer_5_swiglu"
      ]
    },
    "layer_5_up_proj_out_78": {
      "name": "layer_5_up_proj_out_78",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_5_up_proj",
      "consumers": [
        "layer_5_swiglu"
      ]
    },
    "layer_5_swiglu_out_79": {
      "name": "layer_5_swiglu_out_79",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_5_swiglu",
      "consumers": [
        "layer_5_down_proj"
      ]
    },
    "layer_5_down_proj_out_80": {
      "name": "layer_5_down_proj_out_80",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_5_down_proj",
      "consumers": [
        "layer_5_mlp_residual"
      ]
    },
    "layer_5_mlp_residual_out_81": {
      "name": "layer_5_mlp_residual_out_81",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_5_mlp_residual",
      "consumers": [
        "layer_6_attn_norm",
        "layer_6_attn_residual"
      ]
    },
    "layer_6_attn_norm_out_82": {
      "name": "layer_6_attn_norm_out_82",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_6_attn_norm",
      "consumers": [
        "layer_6_q_proj",
        "layer_6_k_proj",
        "layer_6_v_proj"
      ]
    },
    "layer_6_q_proj_out_83": {
      "name": "layer_6_q_proj_out_83",
      "shape": [
        -1,
        -1,
        203
      ],
      "dtype": "float32",
      "node": "layer_6_q_proj",
      "consumers": [
        "layer_6_attn"
      ]
    },
    "layer_6_k_proj_out_84": {
      "name": "layer_6_k_proj_out_84",
      "shape": [
        -1,
        -1,
        29
      ],
      "dtype": "float32",
      "node": "layer_6_k_proj",
      "consumers": [
        "layer_6_attn"
      ]
    },
    "layer_6_v_proj_out_85": {
      "name": "layer_6_v_proj_out_85",
      "shape": [
        -1,
        -1,
        29
      ],
      "dtype": "float32",
      "node": "layer_6_v_proj",
      "consumers": [
        "layer_6_attn"
      ]
    },
    "layer_6_attn_out_86": {
      "name": "layer_6_attn_out_86",
      "shape": [
        -1,
        -1,
        203
      ],
      "dtype": "float32",
      "node": "layer_6_attn",
      "consumers": [
        "layer_6_attn_out_proj"
      ]
    },
    "layer_6_attn_out_proj_out_87": {
      "name": "layer_6_attn_out_proj_out_87",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_6_attn_out_proj",
      "consumers": [
        "layer_6_attn_residual"
      ]
    },
    "layer_6_attn_residual_out_88": {
      "name": "layer_6_attn_residual_out_88",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_6_attn_residual",
      "consumers": [
        "layer_6_mlp_norm",
        "layer_6_mlp_residual"
      ]
    },
    "layer_6_mlp_norm_out_89": {
      "name": "layer_6_mlp_norm_out_89",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_6_mlp_norm",
      "consumers": [
        "layer_6_gate_proj",
        "layer_6_up_proj"
      ]
    },
    "layer_6_gate_proj_out_90": {
      "name": "layer_6_gate_proj_out_90",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_6_gate_proj",
      "consumers": [
        "layer_6_swiglu"
      ]
    },
    "layer_6_up_proj_out_91": {
      "name": "layer_6_up_proj_out_91",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_6_up_proj",
      "consumers": [
        "layer_6_swiglu"
      ]
    },
    "layer_6_swiglu_out_92": {
      "name": "layer_6_swiglu_out_92",
      "shape": [
        -1,
        -1,
        4992
      ],
      "dtype": "float32",
      "node": "layer_6_swiglu",
      "consumers": [
        "layer_6_down_proj"
      ]
    },
    "layer_6_down_proj_out_93": {
      "name": "layer_6_down_proj_out_93",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_6_down_proj",
      "consumers": [
        "layer_6_mlp_residual"
      ]
    },
    "layer_6_mlp_residual_out_94": {
      "name": "layer_6_mlp_residual_out_94",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "layer_6_mlp_residual",
      "consumers": [
        "output_norm"
      ]
    },
    "output_norm_out_95": {
      "name": "output_norm_out_95",
      "shape": [
        -1,
        -1,
        931
      ],
      "dtype": "float32",
      "node": "output_norm",
      "consumers": [
        "output_projection"
      ]
    },
    "output_projection_out_96": {
      "name": "output_projection_out_96",
      "shape": [
        -1,
        -1,
        32000
      ],
      "dtype": "float32",
      "node": "output_projection",
      "consumers": []
    }
  },
  "operations": {
    "token_embeddings": {
      "name": "token_embeddings",
      "type": "embedding",
      "inputs": [
        "input_ids"
      ],
      "outputs": [
        "token_embeddings_out_1"
      ],
      "attributes": {
        "vocab_size": 32000,
        "embedding_dim": 931
      }
    },
    "rope_positional": {
      "name": "rope_positional",
      "type": "rope",
      "inputs": [
        "token_embeddings_out_1"
      ],
      "outputs": [
        "rope_positional_out_2"
      ],
      "attributes": {
        "dim": 29,
        "theta": 10000.0
      }
    },
    "input_norm": {
      "name": "input_norm",
      "type": "rmsnorm",
      "inputs": [
        "rope_positional_out_2"
      ],
      "outputs": [
        "input_norm_out_3"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_0_attn_norm": {
      "name": "layer_0_attn_norm",
      "type": "rmsnorm",
      "inputs": [
        "input_norm_out_3"
      ],
      "outputs": [
        "layer_0_attn_norm_out_4"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_0_q_proj": {
      "name": "layer_0_q_proj",
      "type": "linear",
      "inputs": [
        "layer_0_attn_norm_out_4"
      ],
      "outputs": [
        "layer_0_q_proj_out_5"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 203,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_0_k_proj": {
      "name": "layer_0_k_proj",
      "type": "linear",
      "inputs": [
        "layer_0_attn_norm_out_4"
      ],
      "outputs": [
        "layer_0_k_proj_out_6"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 29,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_0_v_proj": {
      "name": "layer_0_v_proj",
      "type": "linear",
      "inputs": [
        "layer_0_attn_norm_out_4"
      ],
      "outputs": [
        "layer_0_v_proj_out_7"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 29,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_0_attn": {
      "name": "layer_0_attn",
      "type": "multi_head_attention",
      "inputs": [
        "layer_0_q_proj_out_5",
        "layer_0_k_proj_out_6",
        "layer_0_v_proj_out_7"
      ],
      "outputs": [
        "layer_0_attn_out_8"
      ],
      "attributes": {
        "num_heads": 7,
        "num_kv_heads": 1,
        "head_dim": 29,
        "attention_type": "gqa",
        "use_alibi": false
      }
    },
    "layer_0_attn_out_proj": {
      "name": "layer_0_attn_out_proj",
      "type": "linear",
      "inputs": [
        "layer_0_attn_out_8"
      ],
      "outputs": [
        "layer_0_attn_out_proj_out_9"
      ],
      "attributes": {
        "in_features": 203,
        "out_features": 931,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_0_attn_residual": {
      "name": "layer_0_attn_residual",
      "type": "add",
      "inputs": [
        "input_norm_out_3",
        "layer_0_attn_out_proj_out_9"
      ],
      "outputs": [
        "layer_0_attn_residual_out_10"
      ],
      "attributes": {}
    },
    "layer_0_mlp_norm": {
      "name": "layer_0_mlp_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_0_attn_residual_out_10"
      ],
      "outputs": [
        "layer_0_mlp_norm_out_11"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_0_gate_proj": {
      "name": "layer_0_gate_proj",
      "type": "linear",
      "inputs": [
        "layer_0_mlp_norm_out_11"
      ],
      "outputs": [
        "layer_0_gate_proj_out_12"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 4992,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_0_up_proj": {
      "name": "layer_0_up_proj",
      "type": "linear",
      "inputs": [
        "layer_0_mlp_norm_out_11"
      ],
      "outputs": [
        "layer_0_up_proj_out_13"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 4992,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_0_swiglu": {
      "name": "layer_0_swiglu",
      "type": "swiglu",
      "inputs": [
        "layer_0_gate_proj_out_12",
        "layer_0_up_proj_out_13"
      ],
      "outputs": [
        "layer_0_swiglu_out_14"
      ],
      "attributes": {}
    },
    "layer_0_down_proj": {
      "name": "layer_0_down_proj",
      "type": "linear",
      "inputs": [
        "layer_0_swiglu_out_14"
      ],
      "outputs": [
        "layer_0_down_proj_out_15"
      ],
      "attributes": {
        "in_features": 4992,
        "out_features": 931,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_0_mlp_residual": {
      "name": "layer_0_mlp_residual",
      "type": "add",
      "inputs": [
        "layer_0_attn_residual_out_10",
        "layer_0_down_proj_out_15"
      ],
      "outputs": [
        "layer_0_mlp_residual_out_16"
      ],
      "attributes": {}
    },
    "layer_1_attn_norm": {
      "name": "layer_1_attn_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_0_mlp_residual_out_16"
      ],
      "outputs": [
        "layer_1_attn_norm_out_17"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_1_q_proj": {
      "name": "layer_1_q_proj",
      "type": "linear",
      "inputs": [
        "layer_1_attn_norm_out_17"
      ],
      "outputs": [
        "layer_1_q_proj_out_18"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 203,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_1_k_proj": {
      "name": "layer_1_k_proj",
      "type": "linear",
      "inputs": [
        "layer_1_attn_norm_out_17"
      ],
      "outputs": [
        "layer_1_k_proj_out_19"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 29,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_1_v_proj": {
      "name": "layer_1_v_proj",
      "type": "linear",
      "inputs": [
        "layer_1_attn_norm_out_17"
      ],
      "outputs": [
        "layer_1_v_proj_out_20"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 29,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_1_attn": {
      "name": "layer_1_attn",
      "type": "multi_head_attention",
      "inputs": [
        "layer_1_q_proj_out_18",
        "layer_1_k_proj_out_19",
        "layer_1_v_proj_out_20"
      ],
      "outputs": [
        "layer_1_attn_out_21"
      ],
      "attributes": {
        "num_heads": 7,
        "num_kv_heads": 1,
        "head_dim": 29,
        "attention_type": "gqa",
        "use_alibi": false
      }
    },
    "layer_1_attn_out_proj": {
      "name": "layer_1_attn_out_proj",
      "type": "linear",
      "inputs": [
        "layer_1_attn_out_21"
      ],
      "outputs": [
        "layer_1_attn_out_proj_out_22"
      ],
      "attributes": {
        "in_features": 203,
        "out_features": 931,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_1_attn_residual": {
      "name": "layer_1_attn_residual",
      "type": "add",
      "inputs": [
        "layer_0_mlp_residual_out_16",
        "layer_1_attn_out_proj_out_22"
      ],
      "outputs": [
        "layer_1_attn_residual_out_23"
      ],
      "attributes": {}
    },
    "layer_1_mlp_norm": {
      "name": "layer_1_mlp_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_1_attn_residual_out_23"
      ],
      "outputs": [
        "layer_1_mlp_norm_out_24"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_1_gate_proj": {
      "name": "layer_1_gate_proj",
      "type": "linear",
      "inputs": [
        "layer_1_mlp_norm_out_24"
      ],
      "outputs": [
        "layer_1_gate_proj_out_25"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 4992,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_1_up_proj": {
      "name": "layer_1_up_proj",
      "type": "linear",
      "inputs": [
        "layer_1_mlp_norm_out_24"
      ],
      "outputs": [
        "layer_1_up_proj_out_26"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 4992,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_1_swiglu": {
      "name": "layer_1_swiglu",
      "type": "swiglu",
      "inputs": [
        "layer_1_gate_proj_out_25",
        "layer_1_up_proj_out_26"
      ],
      "outputs": [
        "layer_1_swiglu_out_27"
      ],
      "attributes": {}
    },
    "layer_1_down_proj": {
      "name": "layer_1_down_proj",
      "type": "linear",
      "inputs": [
        "layer_1_swiglu_out_27"
      ],
      "outputs": [
        "layer_1_down_proj_out_28"
      ],
      "attributes": {
        "in_features": 4992,
        "out_features": 931,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_1_mlp_residual": {
      "name": "layer_1_mlp_residual",
      "type": "add",
      "inputs": [
        "layer_1_attn_residual_out_23",
        "layer_1_down_proj_out_28"
      ],
      "outputs": [
        "layer_1_mlp_residual_out_29"
      ],
      "attributes": {}
    },
    "layer_2_attn_norm": {
      "name": "layer_2_attn_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_1_mlp_residual_out_29"
      ],
      "outputs": [
        "layer_2_attn_norm_out_30"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_2_q_proj": {
      "name": "layer_2_q_proj",
      "type": "linear",
      "inputs": [
        "layer_2_attn_norm_out_30"
      ],
      "outputs": [
        "layer_2_q_proj_out_31"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 203,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_2_k_proj": {
      "name": "layer_2_k_proj",
      "type": "linear",
      "inputs": [
        "layer_2_attn_norm_out_30"
      ],
      "outputs": [
        "layer_2_k_proj_out_32"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 29,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_2_v_proj": {
      "name": "layer_2_v_proj",
      "type": "linear",
      "inputs": [
        "layer_2_attn_norm_out_30"
      ],
      "outputs": [
        "layer_2_v_proj_out_33"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 29,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_2_attn": {
      "name": "layer_2_attn",
      "type": "multi_head_attention",
      "inputs": [
        "layer_2_q_proj_out_31",
        "layer_2_k_proj_out_32",
        "layer_2_v_proj_out_33"
      ],
      "outputs": [
        "layer_2_attn_out_34"
      ],
      "attributes": {
        "num_heads": 7,
        "num_kv_heads": 1,
        "head_dim": 29,
        "attention_type": "gqa",
        "use_alibi": false
      }
    },
    "layer_2_attn_out_proj": {
      "name": "layer_2_attn_out_proj",
      "type": "linear",
      "inputs": [
        "layer_2_attn_out_34"
      ],
      "outputs": [
        "layer_2_attn_out_proj_out_35"
      ],
      "attributes": {
        "in_features": 203,
        "out_features": 931,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_2_attn_residual": {
      "name": "layer_2_attn_residual",
      "type": "add",
      "inputs": [
        "layer_1_mlp_residual_out_29",
        "layer_2_attn_out_proj_out_35"
      ],
      "outputs": [
        "layer_2_attn_residual_out_36"
      ],
      "attributes": {}
    },
    "layer_2_mlp_norm": {
      "name": "layer_2_mlp_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_2_attn_residual_out_36"
      ],
      "outputs": [
        "layer_2_mlp_norm_out_37"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_2_gate_proj": {
      "name": "layer_2_gate_proj",
      "type": "linear",
      "inputs": [
        "layer_2_mlp_norm_out_37"
      ],
      "outputs": [
        "layer_2_gate_proj_out_38"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 4992,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_2_up_proj": {
      "name": "layer_2_up_proj",
      "type": "linear",
      "inputs": [
        "layer_2_mlp_norm_out_37"
      ],
      "outputs": [
        "layer_2_up_proj_out_39"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 4992,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_2_swiglu": {
      "name": "layer_2_swiglu",
      "type": "swiglu",
      "inputs": [
        "layer_2_gate_proj_out_38",
        "layer_2_up_proj_out_39"
      ],
      "outputs": [
        "layer_2_swiglu_out_40"
      ],
      "attributes": {}
    },
    "layer_2_down_proj": {
      "name": "layer_2_down_proj",
      "type": "linear",
      "inputs": [
        "layer_2_swiglu_out_40"
      ],
      "outputs": [
        "layer_2_down_proj_out_41"
      ],
      "attributes": {
        "in_features": 4992,
        "out_features": 931,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_2_mlp_residual": {
      "name": "layer_2_mlp_residual",
      "type": "add",
      "inputs": [
        "layer_2_attn_residual_out_36",
        "layer_2_down_proj_out_41"
      ],
      "outputs": [
        "layer_2_mlp_residual_out_42"
      ],
      "attributes": {}
    },
    "layer_3_attn_norm": {
      "name": "layer_3_attn_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_2_mlp_residual_out_42"
      ],
      "outputs": [
        "layer_3_attn_norm_out_43"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_3_q_proj": {
      "name": "layer_3_q_proj",
      "type": "linear",
      "inputs": [
        "layer_3_attn_norm_out_43"
      ],
      "outputs": [
        "layer_3_q_proj_out_44"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 203,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_3_k_proj": {
      "name": "layer_3_k_proj",
      "type": "linear",
      "inputs": [
        "layer_3_attn_norm_out_43"
      ],
      "outputs": [
        "layer_3_k_proj_out_45"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 29,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_3_v_proj": {
      "name": "layer_3_v_proj",
      "type": "linear",
      "inputs": [
        "layer_3_attn_norm_out_43"
      ],
      "outputs": [
        "layer_3_v_proj_out_46"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 29,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_3_attn": {
      "name": "layer_3_attn",
      "type": "multi_head_attention",
      "inputs": [
        "layer_3_q_proj_out_44",
        "layer_3_k_proj_out_45",
        "layer_3_v_proj_out_46"
      ],
      "outputs": [
        "layer_3_attn_out_47"
      ],
      "attributes": {
        "num_heads": 7,
        "num_kv_heads": 1,
        "head_dim": 29,
        "attention_type": "gqa",
        "use_alibi": false
      }
    },
    "layer_3_attn_out_proj": {
      "name": "layer_3_attn_out_proj",
      "type": "linear",
      "inputs": [
        "layer_3_attn_out_47"
      ],
      "outputs": [
        "layer_3_attn_out_proj_out_48"
      ],
      "attributes": {
        "in_features": 203,
        "out_features": 931,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_3_attn_residual": {
      "name": "layer_3_attn_residual",
      "type": "add",
      "inputs": [
        "layer_2_mlp_residual_out_42",
        "layer_3_attn_out_proj_out_48"
      ],
      "outputs": [
        "layer_3_attn_residual_out_49"
      ],
      "attributes": {}
    },
    "layer_3_mlp_norm": {
      "name": "layer_3_mlp_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_3_attn_residual_out_49"
      ],
      "outputs": [
        "layer_3_mlp_norm_out_50"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_3_gate_proj": {
      "name": "layer_3_gate_proj",
      "type": "linear",
      "inputs": [
        "layer_3_mlp_norm_out_50"
      ],
      "outputs": [
        "layer_3_gate_proj_out_51"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 4992,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_3_up_proj": {
      "name": "layer_3_up_proj",
      "type": "linear",
      "inputs": [
        "layer_3_mlp_norm_out_50"
      ],
      "outputs": [
        "layer_3_up_proj_out_52"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 4992,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_3_swiglu": {
      "name": "layer_3_swiglu",
      "type": "swiglu",
      "inputs": [
        "layer_3_gate_proj_out_51",
        "layer_3_up_proj_out_52"
      ],
      "outputs": [
        "layer_3_swiglu_out_53"
      ],
      "attributes": {}
    },
    "layer_3_down_proj": {
      "name": "layer_3_down_proj",
      "type": "linear",
      "inputs": [
        "layer_3_swiglu_out_53"
      ],
      "outputs": [
        "layer_3_down_proj_out_54"
      ],
      "attributes": {
        "in_features": 4992,
        "out_features": 931,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_3_mlp_residual": {
      "name": "layer_3_mlp_residual",
      "type": "add",
      "inputs": [
        "layer_3_attn_residual_out_49",
        "layer_3_down_proj_out_54"
      ],
      "outputs": [
        "layer_3_mlp_residual_out_55"
      ],
      "attributes": {}
    },
    "layer_4_attn_norm": {
      "name": "layer_4_attn_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_3_mlp_residual_out_55"
      ],
      "outputs": [
        "layer_4_attn_norm_out_56"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_4_q_proj": {
      "name": "layer_4_q_proj",
      "type": "linear",
      "inputs": [
        "layer_4_attn_norm_out_56"
      ],
      "outputs": [
        "layer_4_q_proj_out_57"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 203,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_4_k_proj": {
      "name": "layer_4_k_proj",
      "type": "linear",
      "inputs": [
        "layer_4_attn_norm_out_56"
      ],
      "outputs": [
        "layer_4_k_proj_out_58"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 29,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_4_v_proj": {
      "name": "layer_4_v_proj",
      "type": "linear",
      "inputs": [
        "layer_4_attn_norm_out_56"
      ],
      "outputs": [
        "layer_4_v_proj_out_59"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 29,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_4_attn": {
      "name": "layer_4_attn",
      "type": "multi_head_attention",
      "inputs": [
        "layer_4_q_proj_out_57",
        "layer_4_k_proj_out_58",
        "layer_4_v_proj_out_59"
      ],
      "outputs": [
        "layer_4_attn_out_60"
      ],
      "attributes": {
        "num_heads": 7,
        "num_kv_heads": 1,
        "head_dim": 29,
        "attention_type": "gqa",
        "use_alibi": false
      }
    },
    "layer_4_attn_out_proj": {
      "name": "layer_4_attn_out_proj",
      "type": "linear",
      "inputs": [
        "layer_4_attn_out_60"
      ],
      "outputs": [
        "layer_4_attn_out_proj_out_61"
      ],
      "attributes": {
        "in_features": 203,
        "out_features": 931,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_4_attn_residual": {
      "name": "layer_4_attn_residual",
      "type": "add",
      "inputs": [
        "layer_3_mlp_residual_out_55",
        "layer_4_attn_out_proj_out_61"
      ],
      "outputs": [
        "layer_4_attn_residual_out_62"
      ],
      "attributes": {}
    },
    "layer_4_mlp_norm": {
      "name": "layer_4_mlp_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_4_attn_residual_out_62"
      ],
      "outputs": [
        "layer_4_mlp_norm_out_63"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_4_gate_proj": {
      "name": "layer_4_gate_proj",
      "type": "linear",
      "inputs": [
        "layer_4_mlp_norm_out_63"
      ],
      "outputs": [
        "layer_4_gate_proj_out_64"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 4992,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_4_up_proj": {
      "name": "layer_4_up_proj",
      "type": "linear",
      "inputs": [
        "layer_4_mlp_norm_out_63"
      ],
      "outputs": [
        "layer_4_up_proj_out_65"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 4992,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_4_swiglu": {
      "name": "layer_4_swiglu",
      "type": "swiglu",
      "inputs": [
        "layer_4_gate_proj_out_64",
        "layer_4_up_proj_out_65"
      ],
      "outputs": [
        "layer_4_swiglu_out_66"
      ],
      "attributes": {}
    },
    "layer_4_down_proj": {
      "name": "layer_4_down_proj",
      "type": "linear",
      "inputs": [
        "layer_4_swiglu_out_66"
      ],
      "outputs": [
        "layer_4_down_proj_out_67"
      ],
      "attributes": {
        "in_features": 4992,
        "out_features": 931,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_4_mlp_residual": {
      "name": "layer_4_mlp_residual",
      "type": "add",
      "inputs": [
        "layer_4_attn_residual_out_62",
        "layer_4_down_proj_out_67"
      ],
      "outputs": [
        "layer_4_mlp_residual_out_68"
      ],
      "attributes": {}
    },
    "layer_5_attn_norm": {
      "name": "layer_5_attn_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_4_mlp_residual_out_68"
      ],
      "outputs": [
        "layer_5_attn_norm_out_69"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_5_q_proj": {
      "name": "layer_5_q_proj",
      "type": "linear",
      "inputs": [
        "layer_5_attn_norm_out_69"
      ],
      "outputs": [
        "layer_5_q_proj_out_70"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 203,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_5_k_proj": {
      "name": "layer_5_k_proj",
      "type": "linear",
      "inputs": [
        "layer_5_attn_norm_out_69"
      ],
      "outputs": [
        "layer_5_k_proj_out_71"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 29,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_5_v_proj": {
      "name": "layer_5_v_proj",
      "type": "linear",
      "inputs": [
        "layer_5_attn_norm_out_69"
      ],
      "outputs": [
        "layer_5_v_proj_out_72"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 29,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_5_attn": {
      "name": "layer_5_attn",
      "type": "multi_head_attention",
      "inputs": [
        "layer_5_q_proj_out_70",
        "layer_5_k_proj_out_71",
        "layer_5_v_proj_out_72"
      ],
      "outputs": [
        "layer_5_attn_out_73"
      ],
      "attributes": {
        "num_heads": 7,
        "num_kv_heads": 1,
        "head_dim": 29,
        "attention_type": "gqa",
        "use_alibi": false
      }
    },
    "layer_5_attn_out_proj": {
      "name": "layer_5_attn_out_proj",
      "type": "linear",
      "inputs": [
        "layer_5_attn_out_73"
      ],
      "outputs": [
        "layer_5_attn_out_proj_out_74"
      ],
      "attributes": {
        "in_features": 203,
        "out_features": 931,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_5_attn_residual": {
      "name": "layer_5_attn_residual",
      "type": "add",
      "inputs": [
        "layer_4_mlp_residual_out_68",
        "layer_5_attn_out_proj_out_74"
      ],
      "outputs": [
        "layer_5_attn_residual_out_75"
      ],
      "attributes": {}
    },
    "layer_5_mlp_norm": {
      "name": "layer_5_mlp_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_5_attn_residual_out_75"
      ],
      "outputs": [
        "layer_5_mlp_norm_out_76"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_5_gate_proj": {
      "name": "layer_5_gate_proj",
      "type": "linear",
      "inputs": [
        "layer_5_mlp_norm_out_76"
      ],
      "outputs": [
        "layer_5_gate_proj_out_77"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 4992,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_5_up_proj": {
      "name": "layer_5_up_proj",
      "type": "linear",
      "inputs": [
        "layer_5_mlp_norm_out_76"
      ],
      "outputs": [
        "layer_5_up_proj_out_78"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 4992,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_5_swiglu": {
      "name": "layer_5_swiglu",
      "type": "swiglu",
      "inputs": [
        "layer_5_gate_proj_out_77",
        "layer_5_up_proj_out_78"
      ],
      "outputs": [
        "layer_5_swiglu_out_79"
      ],
      "attributes": {}
    },
    "layer_5_down_proj": {
      "name": "layer_5_down_proj",
      "type": "linear",
      "inputs": [
        "layer_5_swiglu_out_79"
      ],
      "outputs": [
        "layer_5_down_proj_out_80"
      ],
      "attributes": {
        "in_features": 4992,
        "out_features": 931,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_5_mlp_residual": {
      "name": "layer_5_mlp_residual",
      "type": "add",
      "inputs": [
        "layer_5_attn_residual_out_75",
        "layer_5_down_proj_out_80"
      ],
      "outputs": [
        "layer_5_mlp_residual_out_81"
      ],
      "attributes": {}
    },
    "layer_6_attn_norm": {
      "name": "layer_6_attn_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_5_mlp_residual_out_81"
      ],
      "outputs": [
        "layer_6_attn_norm_out_82"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_6_q_proj": {
      "name": "layer_6_q_proj",
      "type": "linear",
      "inputs": [
        "layer_6_attn_norm_out_82"
      ],
      "outputs": [
        "layer_6_q_proj_out_83"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 203,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_6_k_proj": {
      "name": "layer_6_k_proj",
      "type": "linear",
      "inputs": [
        "layer_6_attn_norm_out_82"
      ],
      "outputs": [
        "layer_6_k_proj_out_84"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 29,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_6_v_proj": {
      "name": "layer_6_v_proj",
      "type": "linear",
      "inputs": [
        "layer_6_attn_norm_out_82"
      ],
      "outputs": [
        "layer_6_v_proj_out_85"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 29,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_6_attn": {
      "name": "layer_6_attn",
      "type": "multi_head_attention",
      "inputs": [
        "layer_6_q_proj_out_83",
        "layer_6_k_proj_out_84",
        "layer_6_v_proj_out_85"
      ],
      "outputs": [
        "layer_6_attn_out_86"
      ],
      "attributes": {
        "num_heads": 7,
        "num_kv_heads": 1,
        "head_dim": 29,
        "attention_type": "gqa",
        "use_alibi": false
      }
    },
    "layer_6_attn_out_proj": {
      "name": "layer_6_attn_out_proj",
      "type": "linear",
      "inputs": [
        "layer_6_attn_out_86"
      ],
      "outputs": [
        "layer_6_attn_out_proj_out_87"
      ],
      "attributes": {
        "in_features": 203,
        "out_features": 931,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_6_attn_residual": {
      "name": "layer_6_attn_residual",
      "type": "add",
      "inputs": [
        "layer_5_mlp_residual_out_81",
        "layer_6_attn_out_proj_out_87"
      ],
      "outputs": [
        "layer_6_attn_residual_out_88"
      ],
      "attributes": {}
    },
    "layer_6_mlp_norm": {
      "name": "layer_6_mlp_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_6_attn_residual_out_88"
      ],
      "outputs": [
        "layer_6_mlp_norm_out_89"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "layer_6_gate_proj": {
      "name": "layer_6_gate_proj",
      "type": "linear",
      "inputs": [
        "layer_6_mlp_norm_out_89"
      ],
      "outputs": [
        "layer_6_gate_proj_out_90"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 4992,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_6_up_proj": {
      "name": "layer_6_up_proj",
      "type": "linear",
      "inputs": [
        "layer_6_mlp_norm_out_89"
      ],
      "outputs": [
        "layer_6_up_proj_out_91"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 4992,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_6_swiglu": {
      "name": "layer_6_swiglu",
      "type": "swiglu",
      "inputs": [
        "layer_6_gate_proj_out_90",
        "layer_6_up_proj_out_91"
      ],
      "outputs": [
        "layer_6_swiglu_out_92"
      ],
      "attributes": {}
    },
    "layer_6_down_proj": {
      "name": "layer_6_down_proj",
      "type": "linear",
      "inputs": [
        "layer_6_swiglu_out_92"
      ],
      "outputs": [
        "layer_6_down_proj_out_93"
      ],
      "attributes": {
        "in_features": 4992,
        "out_features": 931,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_6_mlp_residual": {
      "name": "layer_6_mlp_residual",
      "type": "add",
      "inputs": [
        "layer_6_attn_residual_out_88",
        "layer_6_down_proj_out_93"
      ],
      "outputs": [
        "layer_6_mlp_residual_out_94"
      ],
      "attributes": {}
    },
    "output_norm": {
      "name": "output_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_6_mlp_residual_out_94"
      ],
      "outputs": [
        "output_norm_out_95"
      ],
      "attributes": {
        "normalized_shape": 931,
        "eps": 1e-06
      }
    },
    "output_projection": {
      "name": "output_projection",
      "type": "linear",
      "inputs": [
        "output_norm_out_95"
      ],
      "outputs": [
        "output_projection_out_96"
      ],
      "attributes": {
        "in_features": 931,
        "out_features": 32000,
        "use_bias": false,
        "tie_weight": "token_embeddings.weight"
      }
    }
  },
  "inputs": [
    "input_ids"
  ],
  "outputs": [
    "output_projection_out_96"
  ]
}"""
            IR_DEF = json.loads(IR_JSON)


            # ------------------------------------------------------------- helpers
            def _topological_sort(ir: Dict[str, Any]):
                ops = ir["operations"]
                tensors = ir["tensors"]
                indegree = {name: 0 for name in ops}
                for op_name, op in ops.items():
                    for inp in op["inputs"]:
                        producer = tensors.get(inp, {}).get("node")
                        if producer and producer in indegree:
                            indegree[op_name] += 1
                ready = [name for name, deg in indegree.items() if deg == 0]
                order = []
                while ready:
                    current = ready.pop(0)
                    order.append(current)
                    for out in ops[current]["outputs"]:
                        for consumer in tensors.get(out, {}).get("consumers", []):
                            if consumer in indegree:
                                indegree[consumer] -= 1
                                if indegree[consumer] == 0:
                                    ready.append(consumer)
                if len(order) != len(ops):
                    # Fallback to insertion order if a cycle slipped in
                    return list(ops.keys())
                return order


            class RMSNorm(nn.Module):
                def __init__(self, dim: int, eps: float = 1e-6):
                    super().__init__()
                    self.eps = eps
                    self.weight = nn.Parameter(torch.ones(dim))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    norm_x = x.pow(2).mean(-1, keepdim=True)
                    x_normed = x * torch.rsqrt(norm_x + self.eps)
                    return self.weight * x_normed


            class AttentionOp(nn.Module):
                def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, dropout: float = 0.0):
                    super().__init__()
                    self.num_heads = num_heads
                    self.num_kv_heads = num_kv_heads
                    self.head_dim = head_dim
                    self.dropout = dropout
                    self.num_heads_per_kv = max(1, num_heads // num_kv_heads)

                def forward(self, q, k, v, attention_mask=None):
                    bsz, seq_len, _ = q.shape
                    q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

                    if self.num_kv_heads != self.num_heads:
                        k = k.repeat_interleave(self.num_heads_per_kv, dim=1)
                        v = v.repeat_interleave(self.num_heads_per_kv, dim=1)

                    attn = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attention_mask, dropout_p=self.dropout
                    )
                    attn = attn.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
                    return attn


            def _apply_rope(x: torch.Tensor, theta: float = 10000.0):
                if x.size(-1) % 2 != 0:
                    return x
                half = x.size(-1) // 2
                freqs = torch.arange(half, device=x.device, dtype=x.dtype)
                freqs = theta ** (-freqs / half)
                positions = torch.arange(x.size(1), device=x.device, dtype=x.dtype)
                angles = torch.einsum("i,j->ij", positions, freqs)
                sin, cos = angles.sin(), angles.cos()
                x1, x2 = x[..., :half], x[..., half:]
                rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
                return rotated.reshape_as(x)


            def _activation(kind: str, tensor: torch.Tensor):
                if kind == "relu":
                    return F.relu(tensor)
                if kind == "gelu":
                    return F.gelu(tensor)
                if kind == "silu":
                    return F.silu(tensor)
                if kind == "swiglu":
                    # Caller supplies gate/up split; we only keep hook
                    return tensor
                raise ValueError(f"Unsupported activation: {kind}")


            # -------------------------------------------------------------- model
            class llmc-decoder-only-v1-160m(nn.Module):
                def __init__(self, config: Optional[ModelConfig] = None):
                    super().__init__()
                    self.ir = IR_DEF
                    self.order = _topological_sort(IR_DEF)
                    self.config = config or ModelConfig()
                    self.modules_by_op = nn.ModuleDict()

                    for name, op in self.ir["operations"].items():
                        op_type = op["type"]
                        attrs = op.get("attributes", {})
                        if op_type == "embedding":
                            self.modules_by_op[name] = nn.Embedding(
                                attrs["vocab_size"], attrs["embedding_dim"]
                            )
                        elif op_type == "linear":
                            self.modules_by_op[name] = nn.Linear(
                                attrs["in_features"],
                                attrs["out_features"],
                                bias=attrs.get("use_bias", True),
                            )
                        elif op_type == "rmsnorm":
                            self.modules_by_op[name] = RMSNorm(
                                attrs["normalized_shape"], eps=attrs.get("eps", 1e-6)
                            )
                        elif op_type == "layernorm":
                            self.modules_by_op[name] = nn.LayerNorm(
                                attrs["normalized_shape"], eps=attrs.get("eps", 1e-5)
                            )
                        elif op_type == "multi_head_attention":
                            self.modules_by_op[name] = AttentionOp(
                                attrs["num_heads"],
                                attrs.get("num_kv_heads", attrs["num_heads"]),
                                attrs["head_dim"],
                                dropout=self.config.attention_dropout,
                            )
                        # Parameter-free ops are executed directly in forward

                    # Handle tied weights declared in the IR
                    for name, op in self.ir["operations"].items():
                        attrs = op.get("attributes", {})
                        tie_target = attrs.get("tie_weight")
                        if tie_target and name in self.modules_by_op:
                            target_module = tie_target.split(".")[0]
                            if target_module in self.modules_by_op:
                                self.modules_by_op[name].weight = self.modules_by_op[target_module].weight

                    # Move to configured dtype
                    self.to(dtype=torch.float32)

                def forward(self, **inputs):
                    values: Dict[str, torch.Tensor] = {}

                    # Bind graph inputs
                    for required in self.ir["inputs"]:
                        if required not in inputs:
                            raise ValueError(f"Missing required input '{required}'")
                        values[required] = inputs[required]

                    attention_mask = inputs.get("attention_mask")

                    for op_name in self.order:
                        op = self.ir["operations"][op_name]
                        op_type = op["type"]
                        attrs = op.get("attributes", {})
                        args = [values[i] for i in op.get("inputs", [])]

                        if op_type == "embedding":
                            out = self.modules_by_op[op_name](args[0])
                        elif op_type == "linear":
                            out = self.modules_by_op[op_name](args[0])
                        elif op_type == "rmsnorm" or op_type == "layernorm":
                            out = self.modules_by_op[op_name](args[0])
                        elif op_type == "add":
                            out = args[0] + args[1]
                        elif op_type == "mul":
                            out = args[0] * args[1]
                        elif op_type == "activation":
                            out = _activation(attrs.get("activation", "silu"), args[0])
                        elif op_type == "swiglu":
                            out = F.silu(args[0]) * args[1]
                        elif op_type == "multi_head_attention":
                            out = self.modules_by_op[op_name](args[0], args[1], args[2], attention_mask)
                        elif op_type == "rope":
                            out = _apply_rope(args[0], theta=attrs.get("theta", 10000.0))
                        elif op_type == "softmax":
                            out = F.softmax(args[0], dim=attrs.get("dim", -1))
                        else:
                            raise RuntimeError(f"Unsupported IR op type: {op_type}")

                        # Assume single output per op
                        out_name = op["outputs"][0]
                        values[out_name] = out

                    # Collect graph outputs
                    outputs = {name: values[name] for name in self.ir["outputs"]}
                    if len(outputs) == 1:
                        return next(iter(outputs.values()))
                    return outputs
