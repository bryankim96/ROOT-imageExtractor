"""Load data from ctapipe/MAGIC EventSources and dump to file."""
from dl1_data_handler.writer import DL1DataDumper, DL1DataWriter

import argparse
import logging
import sys
import warnings
import os

import yaml

logger = logging.getLogger(__name__)

# Disable warnings by default
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Read DL1 data via event source into ctapipe containers, \
            then write to a specified output file format."))
    parser.add_argument(
        'runlist',
        help='YAML file containing matched groups of input filenames and \
        output filenames.')
    parser.add_argument(
        '--output_dir', '-o',
        help='Path to directory to create output files. It overwrites the output path found in the runlist.')
    parser.add_argument(
        '--config_file', '-c',
        help='YAML configuration file for settings.')
    parser.add_argument(
        "--debug",
        help="Print all debug logger messages",
        action="store_true")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.INFO)
        logging.getLogger('dl1_data_handler.writer').setLevel(
            logging.INFO)

    runlist = yaml.load(open(args.runlist, 'r'))

    for run in runlist:
        if args.output_dir:
            run['target'] = os.path.join(args.output_dir, os.path.basename(run['target']))

    logger.info("Number of input files in runlist: {}".format(
        len([input_file for run in runlist for input_file in run['inputs']])))
    logger.info("Number of output files requested: {}".format(
        len(runlist)))

    # load options from config file and create DL1 Data Writer object
    if args.config_file:
        logger.info("Reading config file {}...".format(args.config_file))
        config = yaml.load(open(args.config_file, 'r'))

        logger.info("Config file {} loaded.".format(args.config_file))
        logger.info(yaml.dump(config,
                              default_flow_style=False,
                              default_style=''))

        writer_settings = config['Data Writer']['Settings']
        event_src_settings = config['Event Source']['Settings']

        dumper_name = config['Data Dumper']['Type']
        dumper_settings = config['Data Dumper']['Settings']

        # Locate DL1DataDumper subclass
        dumpers = {i.__name__: i for i in DL1DataDumper.__subclasses__()}
        if dumper_name in dumpers:
            data_dumper_class = dumpers[dumper_name]
        else:
            raise ValueError(
                "No subclass of DL1DataDumper: {}".format(dumper_name))

        data_writer = DL1DataWriter(event_source_class=None,
                                    event_source_settings=event_src_settings,
                                    data_dumper_class=data_dumper_class,
                                    data_dumper_settings=dumper_settings,
                                    **writer_settings)
    else:
        logger.info("No config file provided, using default settings")
        data_writer = DL1DataWriter()

    data_writer.process_data(runlist)

if __name__ == '__main__':
    main()
