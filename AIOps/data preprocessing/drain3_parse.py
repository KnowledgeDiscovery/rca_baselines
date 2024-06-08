import ast
import logging
import os
import sys
import glob
import csv
import collections
import re
from typing import Tuple
import yaml
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.persistence_handler import PersistenceHandler
from drain3.template_miner_config import TemplateMinerConfig

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(message)s')


class Drain3Parser(TemplateMiner):
    def __init__(self,
                 persistence_handler: PersistenceHandler = None,
                 config: TemplateMinerConfig = None, extra: dict = {}):
        super().__init__(persistence_handler, config)

        self.out_dir = extra.get("out_dir", "./")
        self.use_match = extra.get("use_match", False)
        self.log_format = extra.get("log_format", "<Time>\t<Content>")
        self.keep_para = extra.get("keep_para", True)
        self.path = extra.get("indir", "./")

    def parse(self, logname: str):
        """Parse file `logname` and write result

        Args:
            logname (str): Name of file name to parse
        """
        self.header, self.regex = self.parse_log_format(self.log_format)
        templates, structure = self.process_file(logname, self.use_match)
        log_file_name = os.path.basename(logname)
        self.write_stats(self.out_dir, log_file_name, templates, structure)

    def process_file(self, in_log_file: str, use_match=False):
        """Parse file `in_log_file` with specified Drain `template_miner` 
            Returns dataset similar to the original drain code to save create _template.csv, _structured.csv file

        Args:
            logname (str): Name of file name to parse
            use_match (bool): Whether to use created log matcher
        Returns:
            Tuple[list, list]: List of extracted log events, List of input data and corresponding log event infromation 
        """
        templates_data = {}
        structure_data = []
        with open(in_log_file) as f:
            lineid = 0
            counter = collections.defaultdict(int)
            for line in f:
                line = line.rstrip()
                try:
                    match = self.regex.search(line)
                    if match is None:
                        logger.warning("Log message %s is not expected format %s. Skipping." %(line, self.regex))
                        continue
                    messages = [match.group(header) for header in self.header]
                    content = match.group("Content")
                except Exception as e:
                    logger.warning("Exception Error: %s, regex: %s, log message: %s" %(repr(e), self.regex, line))
                    continue
                if use_match:
                    """ if using with persistent log """
                    match = self.match(content)
                    if not match:
                        logger.warn(f"No match is found for: {content}")
                        continue
                    cluster_id = match.cluster_id
                    template = match.get_template()
                else:
                    result = self.add_log_message(content)
                    cluster_id = result["cluster_id"]
                    template = result["template_mined"]
                counter[cluster_id] += 1
                template_set = [cluster_id, template, counter[cluster_id]]
                templates_data[template_set[0]] = template_set
                st_data = [lineid] + messages + [cluster_id, template]
                if self.keep_para:
                    params = self.extract_parameters(template, content)
                    if params:
                        parameters = [x.value for x in params]
                    else:
                        parameters = []
                    st_data += [parameters]
                structure_data.append(st_data)
                lineid += 1

        return templates_data, structure_data

    def write_stats(self, out_dir_name: str, file_name: str, templates_data: list, structure_data: list):
        """Write parsed results to files

        Args:
            out_dir_name (str): Name of the directory to save files
            file_name (str): Name of the base file name
            templates_data (list): List of extracted log events
            structure_data (list): List of input data and corresponding log event infromation 
        """
        out_template_file = os.path.join(
            out_dir_name, file_name+"_templates.csv")
        os.makedirs(os.path.dirname(out_template_file), exist_ok=True)
        logger.info(out_template_file)
        with open(out_template_file, "w", newline='') as f:
            sorted_keys = sorted(templates_data.keys(), key=lambda x: int(x))
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["EventId", "EventTemplate", "Occurrence"])
            for k in sorted_keys:
                template = templates_data[k]
                writer.writerow(template)

        out_structured_file = os.path.join(
            out_dir_name, file_name+"_structured.csv")
        logger.info(out_structured_file)
        with open(out_structured_file, "w", newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            header = ["LineId"] + self.header + ["EventId", "EventTemplate"]
            if self.keep_para:
                header += ["ParameterList"]
            writer.writerow(header)
            writer.writerows(structure_data)

    def parse_log_format(self, logformat: str) -> Tuple[list, re.Pattern]:
        """Creates header and regex expression from given logformat
           <Time>\t<Content> -> headers: [Time,Content] regex -> ^(?P<Time>.*?)(?P<\t>.*?)(?P<Content>.*?)$

        Args:
            logformat (str): Format of log data

        Returns:
            Tuple[list,re.Pattern]: List of header of each component, Compiled pattern
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex


def get_persistent(persistent_option: str) -> PersistenceHandler:
    """Return persistent object from given string

    Args:
        persistent_option (str): Persistent string

    Returns:
        PersistentHandler: Created persistent handler
    """
    persistence = None
    if persistent_option:
        method, _, params = persistent_option.partition(":")
        if method == "FILE":
            from drain3.file_persistence import FilePersistence
            persistence = FilePersistence(params)
        if method == "REDIS":
            from drain3.redis_persistence import RedisPersistence
            host, port, db, key, ssl, password = params.split(",")
            persistence = RedisPersistence(host, port, db, password, ssl, key)
    return persistence


def get_val(config: dict, section: str, key: str, fallback):
    """Retrieve value of corresponding key in config or return fallback

    Args:
        config (dict): Dictionary to search for the value
        section (str): Name of section
        key (str): Name of key of value
        fallback (Any): Value used if key does not exist

    Returns:
        Any: Corresponding value or fallback
    """
    if section not in config:
        return fallback
    return config[section].get(key, fallback)


def drain3_config_from_dict(config: TemplateMinerConfig, yaml_config: dict) -> Tuple[TemplateMinerConfig, dict]:
    """Sets parameters values from yaml_config to TemplateMinerConfig

    Args:
        config (TemplateMinerConfig): TemplateMinerConfig to be modified
        yaml_config (dict): Data containing parameters for TemplateMinerConfig

    Returns:
        Tuple[TemplateMinerConfig, dict]: Returns updated TemplateMinerConfig and additional parameter set
    """
    section_profiling = 'PROFILING'
    section_snapshot = 'SNAPSHOT'
    section_drain = 'DRAIN'
    section_masking = 'MASKING'
    config.profiling_enabled = get_val(
        yaml_config, section_profiling, 'enabled', fallback=config.profiling_enabled)
    config.profiling_report_sec = get_val(
        yaml_config, section_profiling, 'report_sec', fallback=config.profiling_report_sec)
    config.snapshot_interval_minutes = get_val(
        yaml_config, section_snapshot, 'snapshot_interval_minutes', fallback=config.snapshot_interval_minutes)
    config.snapshot_compress_state = get_val(
        yaml_config, section_snapshot, 'compress_state', fallback=config.snapshot_compress_state)
    drain_extra_delimiters_str = get_val(yaml_config, section_drain, 'extra_delimiters',
                                         fallback=str(config.drain_extra_delimiters))
    config.drain_extra_delimiters = ast.literal_eval(
        drain_extra_delimiters_str)

    config.drain_sim_th = get_val(
        yaml_config, section_drain, 'sim_th', fallback=config.drain_sim_th)
    config.drain_depth = get_val(
        yaml_config, section_drain, 'depth', fallback=config.drain_depth)
    config.drain_max_children = get_val(
        yaml_config, section_drain, 'max_children', fallback=config.drain_max_children)
    config.drain_max_clusters = get_val(
        yaml_config, section_drain, 'max_clusters', fallback=config.drain_max_clusters)
    config.parametrize_numeric_tokens = get_val(
        yaml_config, section_drain, 'parametrize_numeric_tokens', fallback=config.parametrize_numeric_tokens)

    masking_instructions_str = get_val(
        yaml_config, section_masking, 'masking', fallback=config.masking_instructions)

    config.mask_prefix = get_val(
        yaml_config, section_masking, 'mask_prefix', fallback=config.mask_prefix)
    config.mask_suffix = get_val(
        yaml_config, section_masking, 'mask_suffix', fallback=config.mask_suffix)
    config.parameter_extraction_cache_capacity = get_val(yaml_config, section_masking, 'parameter_extraction_cache_capacity',
                                                         fallback=config.parameter_extraction_cache_capacity)

    masking_instructions = []
    #masking_list = json.loads(masking_instructions_str)
    masking_list = masking_instructions_str
    for mi in masking_list:
        instruction = MaskingInstruction(mi['regex_pattern'], mi['mask_with'])
        masking_instructions.append(instruction)
    config.masking_instructions = masking_instructions

    section_extra = "EXTRA"
    extra = {"use_match": False}
    extra["log_format"] = get_val(
        yaml_config, section_extra, "log_format", fallback="<Time>\t<Content>")
    extra["out_dir"] = get_val(
        yaml_config, section_extra, "out_dir", fallback= "./tmp")

    persistence = None
    section_persist = "PERSISTENCE"
    method = get_val(yaml_config, section_persist, "type", fallback="")
    if  method == "FILE":
        from drain3.file_persistence import FilePersistence
        option = get_val(yaml_config, section_persist, "option", fallback={"file_name": "drain3_persist.bin"})
        persistence = FilePersistence(option.get("file_name"))
    elif method == "REDIS":
        from drain3.redis_persistence import RedisPersistence
        option = get_val(yaml_config, section_persist, "option", {})
        persistence = RedisPersistence(**option)
    elif method == "KAFKA":
        from drain3.kafka_persistence import KafkaPersistence
        option = get_val(yaml_config, section_persist, "option", {})
        persistence = KafkaPersistence(**option)
    if persistence is not None and get_val(yaml_config, section_persist, "mode", fallback="") == "LOAD":
        extra["use_match"] = True

    return persistence, config, extra


def Drain3ParserWrapper(log_format: str, indir: str, outdir: str, depth: int, st: float, rex: str, keep_para: bool, maxChild: int):
    """Create and returns new Drain3 parser 

    Args:
        log_format (str): Foramt of each log line
        indir (str): Name of directory to parse
        outdir (str): Name of directory to store results
        depth (int): Parameter for drain3
        st (float): Parameter for drain3
        rex (str): Not used
        keep_para (bool): Whether to keep list of parameters of each log line
        maxChild (int): Parameter for drain3

    Returns:
        Drain3Parser: Drain3 parser 
    """
    config = TemplateMinerConfig()
    config.drain_depth = depth
    config.drain_sim_th = st
    config.drain_max_children = maxChild
    extra_config = {"use_match": False, "out_dir": outdir,
                    "log_format": log_format, "keep_para": keep_para, "indir": indir}
    return Drain3Parser(None, config, extra_config)


# command: python drain3_parse.py --input_dir ./output/onlineBoutique_case_2 --output_dir ./OnlineBoutique_case_2
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", metavar="INPUT_DIR", help="Location of log data")
    parser.add_argument("-o", "--output_dir", type=str, help="Directory to output csv data. Default: ./drain3_result")
    parser.add_argument("-f", "--format", type=str, help="Log_format. Default: <Time>\t<Content>")
    parser.add_argument("-c", "--config", default=None, help="Drain configuration file.  .ini or .yaml")
    parser.add_argument("-s", "--save", type=str, default=None,
                        help="Persistence config: FILE:<file_path>|REDIS:<host>,<port>,<db>,<key>,<ssl>,<pass> ")
    parser.add_argument("-l", "--load", type=str, default=None,
                        help="Persistence config: FILE:<file_path>|REDIS:<host>,<port>,<db>,<key>,<ssl>,<pass>")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    config_file = args.config or ""
    log_dir = args.input_dir

    # default parameter
    extra_config = {"use_match": False, "out_dir": "./drain3_result", "log_format": "<Time>\t<Content>"}
    persistence = None

    config = TemplateMinerConfig()

    if config_file.endswith(".ini"):
        # use drain3's configuration file
        config.load(config_file)    
    elif config_file.endswith(".yaml") or config_file.endswith(".yml"):
        # use custom yaml configuration file
        _cfg = yaml.safe_load(open(config_file))
        persistence, config, extra_config = drain3_config_from_dict(config, _cfg.get("DRAIN3", {}))
    
    # overwrite if supplied in commandline argument
    if args.output_dir: extra_config["out_dir"] = args.output_dir
    if args.format: extra_config["log_format"] = args.log_format

    if args.save or args.load:
        persistence = get_persistent(persistent_option=args.save or args.load)
    if args.load and persistence:
        extra_config["use_match"] = True
    

    template_miner = Drain3Parser(
        persistence, config=config, extra=extra_config)

    for log_file in glob.glob(os.path.join(log_dir,"*")):
        logger.info(log_file)
        try:
            template_miner.parse(log_file)
        except Exception as e:
            logger.warning("Failed to parse %s due to %s" % (log_file, repr(e)))
