# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Dict, Union, Optional

from absl import app
from absl import flags
from absl import logging
from AF2_pipeline.common import protein
from AF2_pipeline.common import residue_constants
from AF2_pipeline.data import pipeline

from AF2_pipeline.data import templates
from AF2_pipeline.tools import hhsearch
from AF2_pipeline.tools import hmmsearch
from AF2_pipeline.model import config
from AF2_pipeline.model import model
from AF2_pipeline.relax import relax
import numpy as np

from AF2_pipeline.model import data
# Internal import (7716).

logging.set_verbosity(logging.INFO)

flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')
flags.DEFINE_list(
    'is_prokaryote_list', None, 'Optional for multimer system, not used by the '
    'single chain system. This list should contain a boolean for each fasta '
    'specifying true where the target complex is from a prokaryote, and false '
    'where it is not, or where the origin is unknown. These values determine '
    'the pairing method for the MSA.')

flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('jackhmmer_binary_path', shutil.which('jackhmmer'),
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', shutil.which('hhblits'),
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', shutil.which('hhsearch'),
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', shutil.which('hmmsearch'),
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', shutil.which('hmmbuild'),
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('kalign_binary_path', shutil.which('kalign'),
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniclust30_database_path', None, 'Path to the Uniclust30 '
                    'database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', None, 'Path to the Uniprot '
                    'database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'Path to the PDB '
                    'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('db_preset', 'full_dbs',
                  ['full_dbs', 'reduced_dbs'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs)')
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_boolean('use_precomputed_msas', False, 'Whether to read MSAs that '
                     'have been written to disk. WARNING: This will not check '
                     'if the sequence, database or configuration have changed.')

flags.DEFINE_boolean('msa', False, 'whether the msa is pre-made ')
flags.DEFINE_string('hhr_path', None, 'path to the hhr file')
flags.DEFINE_string('msa_path', None, 'path to pre-made msa if any')

FLAGS = flags.FLAGS
MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3

def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')

def main(argv):

    use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
    _check_flag('small_bfd_database_path', 'db_preset',
              should_be_set=use_small_bfd)
    _check_flag('bfd_database_path', 'db_preset',
              should_be_set=not use_small_bfd)
    _check_flag('uniclust30_database_path', 'db_preset',
              should_be_set=not use_small_bfd)

    run_multimer_system = 'multimer' in FLAGS.model_preset
    _check_flag('pdb70_database_path', 'model_preset',
              should_be_set=not run_multimer_system)
    _check_flag('pdb_seqres_database_path', 'model_preset',
                should_be_set=run_multimer_system)
    _check_flag('uniprot_database_path', 'model_preset',
              should_be_set=run_multimer_system)
    for tool_name in (
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
        if not FLAGS[f'{tool_name}_binary_path'].value:
            raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                        'sure it is installed on your system.')

    template_searcher = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        databases=[FLAGS.pdb70_database_path])
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
    data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniclust30_database_path=FLAGS.uniclust30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd,
      use_precomputed_msas=FLAGS.use_precomputed_msas)
    
    model_runners = {}
    num_ensemble = 1
    model_names = config.MODEL_PRESETS[FLAGS.model_preset]
    for model_name in model_names:
      model_config = config.model_config(model_name)
      if run_multimer_system:
        model_config.model.num_ensemble_eval = num_ensemble
      else:
        model_config.data.eval.num_ensemble = num_ensemble
      model_params = data.get_model_haiku_params(
          model_name=model_name, data_dir=FLAGS.data_dir)
      model_runner = model.RunModel(model_config, model_params)
      model_runners[model_name] = model_runner

    logging.info('Have %d models: %s', len(model_runners),
                list(model_runners.keys()))
    random_seed = FLAGS.random_seed
    if random_seed is None:
      random_seed = random.randrange(sys.maxsize // len(model_names))
    logging.info('Using random seed %d for the data pipeline', random_seed)
    timings = {}
    # Get features
    t_0 = time.time()
    feature_dict = data_pipeline.process(
          input_fasta_path=FLAGS.fasta_paths,
          msa_output_dir=FLAGS.output_dir,
          hhsearch_result=FLAGS.hhr_path,
          a3m=FLAGS.msa_path)
    timings['features'] = time.time() - t_0
    unrelaxed_pdbs = {}
    relaxed_pdbs = {}
    ranking_confidences = {}
    fasta_name = os.path.split(FLAGS.fasta_paths)[1].split(".")[1]

    # Run the models.
    num_models = len(model_runners)
    for model_index, (model_name, model_runner) in enumerate(
        model_runners.items()):
      logging.info('Running model %s on %s', model_name, fasta_name)
      t_0 = time.time()
      model_random_seed = model_index + random_seed * num_models
      processed_feature_dict = model_runner.process_features(
          feature_dict, random_seed=model_random_seed)
      timings[f'process_features_{model_name}'] = time.time() - t_0

      ################################################################
      #need to save the processed feauture. Note that this feature dict will be different from the feature dict saved previously.
      # output_dir_feature = os.path.join(output_dir, f'{model_name}_feature.pkl')
      # with open(output_dir_feature, 'wb') as f:
      #   pickle.dump(processed_feature_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
      ################################################################

      t_0 = time.time()
      prediction_result = model_runner.predict(processed_feature_dict,
                                              random_seed=model_random_seed)
      t_diff = time.time() - t_0
      #######################################################################
      #output = {
      #    'single': single_activations,
      #    'pair': pair_activations,
      #    # Crop away template rows such that they are not used in MaskedMsaHead.
      #    'msa': msa_activations[:num_sequences, :, :],
      #    'msa_first_row': msa_activations[0],
      #}
      #saving the results
      #print(prediction_result)
      representation = prediction_result['representations']
      output_dir_model = os.path.join(FLAGS.output_dir, f'{model_name}.npz')
      np.savez(output_dir_model, single=representation['single'],
                                  pair=representation['pair'])
                                  #msa=representation['msa'],
                                  #msa_first_row=representation['msa_first_row'])
      #######################################################################

      timings[f'predict_and_compile_{model_name}'] = t_diff
      logging.info(
          'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
          model_name, fasta_name, t_diff)
      '''
      if benchmark:
        t_0 = time.time()
        model_runner.predict(processed_feature_dict,
                            random_seed=model_random_seed)
        t_diff = time.time() - t_0
        timings[f'predict_benchmark_{model_name}'] = t_diff
        logging.info(
            'Total JAX model %s on %s predict time (excludes compilation time): %.1fs',
            model_name, fasta_name, t_diff)
      '''
      plddt = prediction_result['plddt']
      ranking_confidences[model_name] = prediction_result['ranking_confidence']

      # Save the model outputs.
      #result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
      #with open(result_output_path, 'wb') as f:
      #  pickle.dump(prediction_result, f, protocol=4)

      # Add the predicted LDDT in the b-factor column.
      # Note that higher predicted LDDT value means higher model confidence.
      plddt_b_factors = np.repeat(
          plddt[:, None], residue_constants.atom_type_num, axis=-1)
      unrelaxed_protein = protein.from_prediction(
          features=processed_feature_dict,
          result=prediction_result,
          b_factors=plddt_b_factors,
          remove_leading_feature_dimension=not model_runner.multimer_mode)

      unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
      unrelaxed_pdb_path = os.path.join(FLAGS.output_dir, f'unrelaxed_{model_name}.pdb')
      with open(unrelaxed_pdb_path, 'w') as f:
        f.write(unrelaxed_pdbs[model_name])
      amber_relaxer = None
      if amber_relaxer:
        # Relax the prediction.
        t_0 = time.time()
        relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
        timings[f'relax_{model_name}'] = time.time() - t_0

        relaxed_pdbs[model_name] = relaxed_pdb_str

        # Save the relaxed PDB.
        relaxed_output_path = os.path.join(
            FLAGS.output_dir, f'relaxed_{model_name}.pdb')
        with open(relaxed_output_path, 'w') as f:
          f.write(relaxed_pdb_str)

from MSA_generation import msa_generation
import argparse
if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'data_dir',
      'uniref90_database_path',
      'mgnify_database_path',
      'template_mmcif_dir',
      'max_template_date',
      'obsolete_pdbs_path',
  ])
  msa_generation(FLAGS.hhblits_binary_path,\
                FLAGS.hhsearch_binary_path,\
                FLAGS.fasta_paths,\
                FLAGS.output_dir)
  app.run(main)

