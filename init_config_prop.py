# -*- coding: utf-8 *-*

import os
import argparse


def main(args):
    
    with open(args.config_prop_file) as rf:
        config_prop_lines = [line.strip() for line in rf.readlines()]
    
    # Line 9
    config_prop_lines[8] = config_prop_lines[8].split('=')[0] + '=' + os.path.join(args.dst, '2019AB', 'META')
    # Line 17
    config_prop_lines[16] = config_prop_lines[16].split('=')[0] + '=' + os.path.join(args.dst, '2019AB', 'META')
    # Line 19
    config_prop_lines[18] = config_prop_lines[18].split('=')[0] + '=' + os.path.join(args.src, 'config', '2019AB', 'user.c.prop')
    # Line 25
    config_prop_lines[24] = config_prop_lines[24].split('=')[0] + '=' + args.src 
    # Line 32
    config_prop_lines[31] = config_prop_lines[31].split('=')[0] + '=' + args.dst
    # Line 38
    config_prop_lines[37] = config_prop_lines[37].split('=')[0] + '=' + args.src
    
    with open(args.config_prop_file, 'w') as wf:
        for line in config_prop_lines:
            wf.write(line + '\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config_prop_file", action="store", required=True, type=str,
        help="Path to config.prop file."
    )
    parser.add_argument(
        "--src", action="store", required=True, type=str,
        help="Metamorphosys installation source path."
    )
    parser.add_argument(
        "--dst", action="store", required=True, type=str,
        help="Metamorphosys installation destination path."
    )
    
    args = parser.parse_args()
    
    import pprint
    pprint.pprint(vars(args))
    
    main(args)
