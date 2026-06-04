# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,  # To do quick test. Trains/test on small subset of dataset, and # of epochs

        finetune_mode=True,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='3_1_dev_split0_multiaccdoa_foa_model.h5',

        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir='../DCASE2024_SELD_dataset/',

        # OUTPUT PATHS
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir='../DCASE2024_SELD_dataset/seld_feat_label/',

        model_dir='models',  # Dumps the trained models and training curves in this folder
        dcase_output_dir='results',  # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',  # 'foa' - ambisonic or 'mic' - microphone signals

        # FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite=False,  # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,

        # MODEL TYPE
        modality='audio',  # 'audio' or 'audio_visual'
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=50,    # Feature sequence length
        batch_size=128,              # Batch size
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_heads=8,
        nb_self_attn_layers=2,
        nb_transformer_layers=2,

        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,  # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=250,  # Train for maximum epochs
        lr=1e-3,

        # METRIC
        average='macro',                 # Supports 'micro': sample-wise average and 'macro': class-wise average,
        segment_based_metrics=False,     # If True, uses segment-based metrics, else uses frame-based metrics
        evaluate_distance=True,          # If True, computes distance errors and apply distance threshold to the detections
        lad_doa_thresh=20,               # DOA error threshold for computing the detection metrics
        lad_dist_thresh=float('inf'),    # Absolute distance error threshold for computing the detection metrics
        lad_reldist_thresh=float('1'),  # Relative distance error threshold for computing the detection metrics
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '100':
        # SSL-research reproduce config: FOA + multi-ACCDOA, finetune from synthetic ckpt,
        # 4GB VRAM friendly (batch=32), 60 epochs (close to convergence based on official patience).
        print("REPRO-FOA-MA-100ep\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60

    elif argv == '101':
        # Smoke variant of 100: 2 epochs, batch=32, no preheat patience kill.
        print("REPRO-FOA-MA-SMOKE\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 2

    elif argv == '102':
        # MIC + multi-ACCDOA reproduce (uses GCC features per official baseline).
        print("REPRO-MIC-GCC-MA-60ep\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60

    elif argv == '110':
        # MIC + GCC + multi-ACCDOA + GCA full (geometry-aware channel attention,
        # geometry_bias=True). Modern-base experimental cell, finetuned from
        # the official synthetic-pretrained MIC checkpoint (non-strict load
        # skips GCA layers, which keep their random init).
        print("MIC-GCC-MA-60ep + GCA full (geometry_bias=True), finetune from synthetic\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = True
        params['gca_geometry_bias'] = True
        params['gca_n_mics'] = 4
        params['gca_embed_dim'] = 16

    elif argv == '111':
        # MIC + GCC + multi-ACCDOA + GCA no_geom (channel attention without
        # geometry bias). The no_geom control for GCA on modern base.
        print("MIC-GCC-MA-60ep + GCA no_geom (geometry_bias=False), finetune from synthetic\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = True
        params['gca_geometry_bias'] = False
        params['gca_n_mics'] = 4
        params['gca_embed_dim'] = 16

    elif argv == '112':
        # MIC + GCC + multi-ACCDOA, no GCA. Matched control with the same
        # synthetic finetune init as 110/111 -- isolates the GCA contribution.
        # Equivalent to task 102 modulo the job-id naming scheme.
        print("MIC-GCC-MA-60ep no-GCA, finetune from synthetic (matched control)\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False

    elif argv == '113':
        # MIC + GCC + multi-ACCDOA + Vanilla SE-block on the full 10-channel
        # input. The "channel-attention-without-attention-machinery" control:
        # gates each of the 10 input channels with a 2-layer sigmoid MLP after
        # global pooling, no Q/K/V, no per-mic structure, no geometry. Lets
        # us disentangle (i) "channel attention helps in general" from
        # (ii) "per-mic attention with geometry helps" (110, 111).
        print("MIC-GCC-MA-60ep + Vanilla SE-block (channel attn on all 10 ch), finetune\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['use_se_block'] = True
        params['se_block_reduction'] = 2

    elif argv == '120':
        # MIC + GCC + multi-ACCDOA + GCA full at 50% of dev-train data.
        # Path-C data-fraction sweep, paired with 121 (no-GCA, same fraction).
        # 60 epochs, finetune from synthetic. Uses train_data_fraction.
        print("MIC-GCC-MA-60ep + GCA full @ 50% train, finetune from synthetic\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = True
        params['gca_geometry_bias'] = True
        params['gca_n_mics'] = 4
        params['gca_embed_dim'] = 16
        params['train_data_fraction'] = 0.50

    elif argv == '121':
        # MIC + GCC + multi-ACCDOA, no GCA at 50% dev-train data (paired with 120).
        print("MIC-GCC-MA-60ep no-GCA @ 50% train, finetune from synthetic\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['train_data_fraction'] = 0.50

    elif argv == '122':
        # MIC + GCC + multi-ACCDOA + GCA full at 25% dev-train data.
        print("MIC-GCC-MA-60ep + GCA full @ 25% train, finetune from synthetic\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = True
        params['gca_geometry_bias'] = True
        params['gca_n_mics'] = 4
        params['gca_embed_dim'] = 16
        params['train_data_fraction'] = 0.25

    elif argv == '123':
        # MIC + GCC + multi-ACCDOA, no GCA at 25% dev-train data.
        print("MIC-GCC-MA-60ep no-GCA @ 25% train, finetune from synthetic\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['train_data_fraction'] = 0.25

    elif argv == '130':
        # FOA + multi-ACCDDOA + GCA full (geometry over W/X/Y/Z ambisonic
        # channels). Cross-modality counterpart of task 110. Finetune from the
        # synthetic FOA ckpt (3_1_*) so we share the official initialization
        # with task 100.
        print("FOA-MA-60ep + GCA full (FOA-modality, geometry_bias=True), finetune from synthetic\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = True
        params['gca_geometry_bias'] = True
        params['gca_n_mics'] = 4
        params['gca_embed_dim'] = 16
        params['gca_modality'] = 'foa'

    elif argv == '131':
        # FOA + multi-ACCDDOA + GCA no_geom (channel attention without
        # ambisonic-direction bias). The no_geom control for task 130.
        print("FOA-MA-60ep + GCA no_geom (FOA-modality, geometry_bias=False), finetune from synthetic\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = True
        params['gca_geometry_bias'] = False
        params['gca_n_mics'] = 4
        params['gca_embed_dim'] = 16
        params['gca_modality'] = 'foa'

    elif argv == '140':
        # MIC + multi-ACCDDOA + Transformer-only temporal stack (NO GRU,
        # 4x TransformerEncoder layers with d=128). No GCA. Architecture
        # control for the cross-architecture replication of the GCA finding.
        # Loaded non-strictly from synthetic ckpt (Conv weights transfer,
        # TransformerEncoder layers init from scratch).
        print("MIC-MA-60ep + Transformer-only temporal stack (no GRU, no GCA)\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['temporal_arch'] = 'transformer'
        params['nb_transformer_blocks'] = 4

    elif argv == '141':
        # MIC + Transformer-only + GCA full (geometry_bias=True). Tests
        # whether the geometry prior still hurts in a non-CRNN backbone.
        print("MIC-MA-60ep + Transformer-only + GCA full (geometry_bias=True)\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = True
        params['gca_geometry_bias'] = True
        params['gca_n_mics'] = 4
        params['gca_embed_dim'] = 16
        params['temporal_arch'] = 'transformer'
        params['nb_transformer_blocks'] = 4

    elif argv == '142':
        # MIC + Transformer-only + GCA no_geom. Channel-attention control
        # for the Transformer-only variant.
        print("MIC-MA-60ep + Transformer-only + GCA no_geom (geometry_bias=False)\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = True
        params['gca_geometry_bias'] = False
        params['gca_n_mics'] = 4
        params['gca_embed_dim'] = 16
        params['temporal_arch'] = 'transformer'
        params['nb_transformer_blocks'] = 4

    elif argv == '150':
        # FOA + Transformer-only + no GCA. Architecture/modality control
        # for the 2x2 (modality x architecture) dissociation table.
        print("FOA-MA-60ep + Transformer-only (no GRU, no GCA), finetune from synthetic\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['temporal_arch'] = 'transformer'
        params['nb_transformer_blocks'] = 4

    elif argv == '151':
        # FOA + Transformer-only + GCA full (geometry over W/X/Y/Z).
        # Cross-modality counterpart of task 141; closes the 2x2 dissociation.
        print("FOA-MA-60ep + Transformer-only + GCA full (FOA-modality)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = True
        params['gca_geometry_bias'] = True
        params['gca_n_mics'] = 4
        params['gca_embed_dim'] = 16
        params['gca_modality'] = 'foa'
        params['temporal_arch'] = 'transformer'
        params['nb_transformer_blocks'] = 4

    elif argv == '152':
        # FOA + Transformer-only + GCA no_geom. Channel-attention control
        # for task 151 (no ambisonic-direction bias).
        print("FOA-MA-60ep + Transformer-only + GCA no_geom (FOA-modality)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = True
        params['gca_geometry_bias'] = False
        params['gca_n_mics'] = 4
        params['gca_embed_dim'] = 16
        params['gca_modality'] = 'foa'
        params['temporal_arch'] = 'transformer'
        params['nb_transformer_blocks'] = 4

    elif argv == '160':
        # MIC + Conformer temporal stack (no GRU), no GCA. Third backbone on
        # the architecture axis (conv/attention hybrid).
        print("MIC-MA-60ep + Conformer temporal stack (no GRU, no GCA)\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['temporal_arch'] = 'conformer'
        params['nb_conformer_blocks'] = 4

    elif argv == '161':
        # MIC + Conformer + GCA full (geometry_bias=True).
        print("MIC-MA-60ep + Conformer + GCA full (geometry_bias=True)\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = True
        params['gca_geometry_bias'] = True
        params['gca_n_mics'] = 4
        params['gca_embed_dim'] = 16
        params['temporal_arch'] = 'conformer'
        params['nb_conformer_blocks'] = 4

    elif argv == '162':
        # MIC + Conformer + GCA no_geom. Channel-attention control.
        print("MIC-MA-60ep + Conformer + GCA no_geom (geometry_bias=False)\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = True
        params['gca_geometry_bias'] = False
        params['gca_n_mics'] = 4
        params['gca_embed_dim'] = 16
        params['temporal_arch'] = 'conformer'
        params['nb_conformer_blocks'] = 4

    elif argv == '170':
        # FOA + Conformer + no GCA.
        print("FOA-MA-60ep + Conformer (no GRU, no GCA), finetune from synthetic\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['temporal_arch'] = 'conformer'
        params['nb_conformer_blocks'] = 4

    elif argv == '171':
        # FOA + Conformer + GCA full (geometry over W/X/Y/Z).
        print("FOA-MA-60ep + Conformer + GCA full (FOA-modality)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = True
        params['gca_geometry_bias'] = True
        params['gca_n_mics'] = 4
        params['gca_embed_dim'] = 16
        params['gca_modality'] = 'foa'
        params['temporal_arch'] = 'conformer'
        params['nb_conformer_blocks'] = 4

    elif argv == '172':
        # FOA + Conformer + GCA no_geom.
        print("FOA-MA-60ep + Conformer + GCA no_geom (FOA-modality)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = True
        params['gca_geometry_bias'] = False
        params['gca_n_mics'] = 4
        params['gca_embed_dim'] = 16
        params['gca_modality'] = 'foa'
        params['temporal_arch'] = 'conformer'
        params['nb_conformer_blocks'] = 4

    elif argv == '180':
        # FOA + CRNN + convbias geometry injection, FULL. Second injection
        # mechanism (geometry as a learned per-filter conv bias) on the
        # recurrent backbone where GCA HELPS (cf. task 130). Tests whether the
        # architecture-graded effect generalizes beyond GCA-style injection.
        print("FOA-MA-60ep + CRNN + convbias FULL (geometry as conv-feature bias)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['geometry_mode'] = 'convbias'
        params['gca_geometry_bias'] = True
        params['gca_modality'] = 'foa'

    elif argv == '181':
        # FOA + CRNN + convbias no_geom (zeros through the same projection,
        # matched parameter count). The no_geom control for task 180.
        print("FOA-MA-60ep + CRNN + convbias NO_GEOM (matched-capacity control)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['geometry_mode'] = 'convbias'
        params['gca_geometry_bias'] = False
        params['gca_modality'] = 'foa'

    elif argv == '182':
        # MIC + Transformer + convbias geometry injection, FULL. Second
        # injection mechanism on the pure-attention backbone where GCA HURTS
        # (cf. task 141). With 180/181 this closes a 2-cell cross-injection
        # robustness check of the helps->harms ordering.
        print("MIC-MA-60ep + Transformer + convbias FULL (geometry as conv-feature bias)\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['geometry_mode'] = 'convbias'
        params['gca_geometry_bias'] = True
        params['gca_modality'] = 'mic'
        params['temporal_arch'] = 'transformer'
        params['nb_transformer_blocks'] = 4

    elif argv == '183':
        # MIC + Transformer + convbias no_geom. The no_geom control for task 182.
        print("MIC-MA-60ep + Transformer + convbias NO_GEOM (matched-capacity control)\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['geometry_mode'] = 'convbias'
        params['gca_geometry_bias'] = False
        params['gca_modality'] = 'mic'
        params['temporal_arch'] = 'transformer'
        params['nb_transformer_blocks'] = 4

    elif argv == '184':
        # FOA + Conformer + convbias geometry injection, FULL. The MIDDLE point
        # of the FOA architecture axis under the second injection mechanism
        # (cf. GCA task 171). With 180/181 (CRNN) and 186/187 (Transformer) this
        # gives a clean FOA-only helps->harms ordering under convbias, directly
        # comparable to the GCA ordering on the same modality.
        print("FOA-MA-60ep + Conformer + convbias FULL (geometry as conv-feature bias)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['geometry_mode'] = 'convbias'
        params['gca_geometry_bias'] = True
        params['gca_modality'] = 'foa'
        params['temporal_arch'] = 'conformer'
        params['nb_conformer_blocks'] = 4

    elif argv == '185':
        # FOA + Conformer + convbias no_geom. The no_geom control for task 184.
        print("FOA-MA-60ep + Conformer + convbias NO_GEOM (matched-capacity control)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['geometry_mode'] = 'convbias'
        params['gca_geometry_bias'] = False
        params['gca_modality'] = 'foa'
        params['temporal_arch'] = 'conformer'
        params['nb_conformer_blocks'] = 4

    elif argv == '186':
        # FOA + Transformer + convbias geometry injection, FULL. The pure-attention
        # end of the FOA architecture axis under the second injection mechanism
        # (cf. GCA task 151). Pairs with 180/181 (CRNN) to give a modality-fixed,
        # architecture-only contrast that cleanly attributes the effect to
        # architecture rather than modality.
        print("FOA-MA-60ep + Transformer + convbias FULL (geometry as conv-feature bias)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['geometry_mode'] = 'convbias'
        params['gca_geometry_bias'] = True
        params['gca_modality'] = 'foa'
        params['temporal_arch'] = 'transformer'
        params['nb_transformer_blocks'] = 4

    elif argv == '187':
        # FOA + Transformer + convbias no_geom. The no_geom control for task 186.
        print("FOA-MA-60ep + Transformer + convbias NO_GEOM (matched-capacity control)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_1_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 32
        params['nb_epochs'] = 60
        params['use_gca'] = False
        params['geometry_mode'] = 'convbias'
        params['gca_geometry_bias'] = False
        params['gca_modality'] = 'foa'
        params['temporal_arch'] = 'transformer'
        params['nb_transformer_blocks'] = 4

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]  # CNN time pooling
    params['patience'] = int(params['nb_epochs'])  # Stop training if patience is reached
    params['model_dir'] = params['model_dir'] + '_' + params['modality']
    params['dcase_output_dir'] = params['dcase_output_dir'] + '_' + params['modality']

    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2023' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2024' in params['dataset_dir']:
        params['unique_classes'] = 13

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
