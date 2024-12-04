import sys
from pathlib import Path
sys.path.append("../notebooks")

import _model_variations as mv


mv._main(model_id = "0",
         rseed = 13,
         model_name = "model23_ll_lp",
         model_data = None,
         num_warmup = 2_000,
         num_samples = 10_000,
         include_potential_energy = True,
         include_mean_accept_prob = False,
         include_extra_fields = True,
         progress_bar = True,
         save_dir = "." ,
         initial_position = None,
         save_warmup = True,
         load_warmup = True,
         jax_profile = False)
         
