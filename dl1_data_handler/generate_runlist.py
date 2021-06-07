"""Generate runlist to process simtel or root files."""
import yaml
import argparse
import os
import glob
import re


# gamma_20deg_0deg_run26640___cta-prod3-lapalma-2147m-LaPalma-SCT_cone10.simtel.gz

def parse_simtel_filename(filename):
    """Extract run parameters from a simtel.gz filename.
    Parameters
    ----------
    filename : list
        A filename formatted as [particle_type]_[ze]deg_[az]deg_run[run_number]
        ___[production info].simtel.gz
    Returns
    -------
    int
        Run/observation number
    str
        Type of the event primary particle (i.e. gamma, proton)
    int
        Run zenith angle in degrees
    int
        Run azimuth angle in degrees
    str
        A string containing other production-related descriptors.
    str
        N/A.

    """
    basename = os.path.basename(filename).replace('.simtel.gz', '')
    foutputs = re.split(
        '___', basename)
    prod_info = foutputs[1]
    soutputs = re.split(
        '_', foutputs[0])
    particle_type, ze, az, run_number = soutputs
    identifiers = [particle_type, ze, az, prod_info]

    identifiers[1] = int(re.sub('deg$', '', identifiers[1]))
    identifiers[2] = int(re.sub('deg$', '', identifiers[2]))
    run_number = int(re.sub('^run', '', run_number))

    return run_number, identifiers, False

def parse_root_filename(filename):
    """Extract run parameters from a root filename.
    Parameters
    ----------
    filename : list
        A filename formatted as MARS file
    Returns
    -------
    int
        Run/observation number
    array
        Identifiers depending on the origin of the input (simu/real)
    """

    basename = os.path.basename(filename).replace('.root', '')
    soutputs = re.split(
       '_', basename)

    if soutputs[0].isnumeric():
        date, run_number, alpha, source = soutputs
        identifiers = [date, alpha, source]
    else:
        particle_type, info, num, run_number, alpha, w = soutputs
        identifiers = [particle_type, info, num, alpha, w]

    run_number = int(re.sub('^run', '', run_number))

    return run_number, identifiers

def parse_filename(filename, filename_type):
    return parse_root_filename(filename) if filename_type == "root" else parse_root_filename(filename)

def main():

    parser = argparse.ArgumentParser(
        description=("Generate a runlist automatically from simtel.gz or root \
                     files following the naming convention \
                     [particle_type]_[ze]deg_[az]deg_run[run_number]___ \
                     [production info].simtel.gz or the naming convention\
                     of the MAGIC-MARS superstar files, respectively."))
    parser.add_argument('file_dir',
                        help=("path to directory containing all simtel files"))
    parser.add_argument('--num_inputs_per_run', '-n',
                        help='number of input files between each run',
                        default=10)
    parser.add_argument('--output_file_name', '-f',
                        help='filepath/name of runlist file without a postfix',
                        default="./runlist")
    parser.add_argument('--output_dir', '-o',
                        help='path where to save generated files. By default, the input directory is used.',
                        default=None,
                       )
    args = parser.parse_args()

    runlist = []
    dl_runlist = []

    abs_file_dir = os.path.abspath(args.file_dir)
    filenames = glob.glob(os.path.join(abs_file_dir, "*.simtel.gz"))
    filename_type = "simtel"
    if filenames == []:
        filenames = glob.glob(os.path.join(abs_file_dir, "*.root"))
        filename_type = "root"

    filename_groups = {}

    output_dir = abs_file_dir if args.output_dir is None else os.path.abspath(args.output_dir)

    # Separate files by identifiers:
    # simtel: particle type, ze, az, processing info, cone
    # root simu: particle_type, info, num, alpha, w
    # root real: date, alpha, source
    for filename in filenames:
        identifiers = tuple(parse_filename(filename, filename_type)[1])
        if identifiers not in filename_groups:
            filename_groups[identifiers] = []
        filename_groups[identifiers].append(filename)

    for key, list in filename_groups.items():
        # Within each group, sort files by run number
        list.sort(key=lambda x: parse_filename(x, filename_type)[0])

        inputs = []
        start_run_number = 0
        for i, filename in enumerate(list):
            run_number = parse_filename(filename, filename_type)[0]
            if len(inputs) == 0:
                start_run_number = run_number
            inputs.append(filename)

            if len(inputs) >= int(args.num_inputs_per_run) or i == (
                    len(list) - 1):
                if filename_type == "root":
                    if identifiers[0].isnumeric():
                        date, alpha, source = key
                        target_filename = "{}/{}_runs{}_{}_{}.h5".format(
                            output_dir,
                            date,
                            '{}-{}'.format(start_run_number, run_number),
                            alpha,
                            source)
                    else:
                        particle_type, info, num, alpha, w = key
                        target_filename = "{}/{}_{}_{}_runs{}_{}_{}.h5".format(
                            output_dir,
                            particle_type,
                            info,
                            num,
                            '{}-{}'.format(start_run_number, run_number),
                            alpha,
                            w)
                elif filename_type == "simtel":
                    particle_type, ze, az, prod_info = key

                    target_filename = "{}/{}_{}deg_{}deg_runs{}___{}.h5".format(
                        output_dir,
                        particle_type,
                        ze,
                        az,
                        '{}-{}'.format(start_run_number, run_number),
                        prod_info)

                runlist.append({
                    'inputs': inputs,
                    'target': target_filename
                })

                dl_runlist.append(target_filename)

                inputs = []

    # dump to file
    stream = open(f'{args.output_file_name}.yml', 'w')
    yaml.dump(runlist, stream)

    with open(f'{args.output_file_name}.txt', 'w') as f:
        for item in dl_runlist:
            f.write("%s\n" % item)

if __name__ == "__main__":
    main()
