import regex as re
import os
import numpy as np
import pandas as pd
from datetime import datetime
import json
from .gpt_query import (
    post_process_template,
    query_template_from_gpt,
    query_template_from_gpt_with_check,
)
from .parsing_cache import ParsingCache
from .post_process import correct_single_template
from .utils import load_pickle, save_pickle, load_tuple_list, cache_to_file
from tqdm import tqdm


def save_results_to_csv(log_file, template_file, cache_file, output_file, output_template_file):
    with open(log_file, 'r') as f:
        lines_a = f.readlines()
    with open(template_file, 'r') as f:
        lines_b = f.readlines()
    cache = load_pickle(cache_file)
    total_len = len(lines_a)
    lineids = range(1, total_len + 1)
    eventids = [''] * total_len
    contents = [''] * total_len
    event_templates = [''] * total_len
    templates_set = []
    print("start writing log structured csv.")
    for (line_a, line_b) in zip(lines_a, lines_b):
        idx_a, str_a = line_a.strip().split(' ', 1)
        idx_b, str_b = line_b.strip().split(' ', 1)
        idx_a, idx_b = int(idx_a), int(idx_b)
        str_b = cache.template_list[int(str_b)]
        if idx_a != idx_b:
            print(f"Error in line: {idx_a} {idx_b}")
            return
        if str_b in templates_set:
            template_id = templates_set.index(str_b) + 1
        else:
            templates_set.append(str_b)
            template_id = len(templates_set)
        # print(idx_a)
        contents[idx_a] = str_a
        event_templates[idx_a] = str_b
        eventids[idx_a] = f"E{template_id}"
    print("end writing log structured csv.")

    df = pd.DataFrame({'LineId': lineids, 'EventId': eventids, 'Content': contents, 'EventTemplate': event_templates})
    df.to_csv(output_file, index=False)

    template_ids = [f"E{i+1}" for i in range(len(templates_set))]
    df = pd.DataFrame({'EventId': template_ids, 'EventTemplate': templates_set})
    df.to_csv(output_template_file, index=False)


def load_regs():
    regs_common = []
    with open(os.path.join(os.path.dirname(__file__), "common.json"), "r") as fr:
        dic = json.load(fr)
    
    patterns = dic['COMMON']['regex']
    for pattern in patterns:
        regs_common.append(re.compile(pattern))
    return regs_common


class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', rex=[],
                 data_type='2k', model="gpt-3.5-turbo", run_id=None):
        self.path = indir
        self.outdir = outdir
        self.df_log = None
        self.log_format = log_format
        self.data_type = data_type
        self.model = model
        self.run_id = run_id

    def parse(self, logName):
        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName
        dataset_name = logName.split('_')[0]
        log_file_base = os.path.splitext(logName)[0]
        run_suffix = f"_{self.run_id}" if self.run_id else ""
        output_path = os.path.join(f"../../temp/lilac_temp_{self.data_type}_{self.model}{run_suffix}", dataset_name)
        evaluation_path = self.outdir
        if os.path.exists(os.path.join(evaluation_path, f"{log_file_base}.log_structured.csv")):
            print(f"{dataset_name} already exists.")
            return
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)
        self.load_data()

        regs_common = load_regs()

        cached_tree = os.path.join(output_path, "cached_tree.pkl")
        cached_log = os.path.join(output_path, "cached_log.txt")
        cached_template = os.path.join(output_path, "cached_template.txt")
        if os.path.exists(cached_tree) and os.path.exists(cached_log) and os.path.exists(cached_template):
            cache = load_pickle(cached_tree)
            log_messages = load_tuple_list(cached_log)
            log_templates = load_tuple_list(cached_template)
            idx = log_messages[-1][1]
        else:
            log_messages = []
            log_templates = []
            cache = ParsingCache()
            idx = 0

        num_query = 0
        total_line = len(self.df_log)
        cache_step = total_line // 5
        if idx + 1 < total_line:
            for log in list(self.df_log[idx:]['Content']):
                flag = self.process_log(cache, [log], log_messages, log_templates, idx, regs_common, total_line)
                idx += 1
                if flag:
                    num_query += 1
                    print("Query times: ", num_query)
                if idx % cache_step == 0:
                    print("Finished processing line: ", idx)
                    cache_to_file(log_messages, cached_log)
                    cache_to_file(log_templates, cached_template)
                    save_pickle(cache, cached_tree)                
        if num_query > 0:
            print("Total query: ", num_query)

        cache_to_file(log_messages, cached_log)
        cache_to_file(log_templates, cached_template)
        save_pickle(cache, cached_tree)
        save_results_to_csv(cached_log, cached_template, cached_tree,
                            os.path.join(evaluation_path, f"{log_file_base}.log_structured.csv"),
                            os.path.join(evaluation_path, f"{log_file_base}.log_templates.csv"))

        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))


    def query_ambiguity_candidate(self, log, regs_common):
        template, generated = query_template_from_gpt(log, [], self.model)
        if not generated:
            return None

        template, valid = post_process_template(template, regs_common, log)
        if not valid:
            return None

        check_tree = ParsingCache()
        check_tree.add_templates(template)
        if check_tree.match_event(log)[0] == "NoMatch":
            return None
        return template

    def process_log(self, cache, logs, log_messages, log_templates, idx, regs_common, total_line):
        new_template = None
        for log in logs:
            results = cache.match_event_with_ambiguity(log)
            if results[0] == "NoMatch":
                print("===========================================")
                print(f"Line-{idx}/{total_line}: No match. {log}")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
                new_template, normal = query_template_from_gpt_with_check(log, regs_common, [], self.model)
                # new_template = post_process_template(new_template)
                print("queried_new_template: ", new_template)
                template_id = cache.add_templates(
                    new_template,
                    normal,
                    results[2],
                    source_log=log,
                )
                log_messages.append((log, idx))
                log_templates.append((template_id, idx))
                # results = cache.match_event(log)
                # log_templates.append((results[log][0], idx))
                # print(results)
                print("===========================================")
                return True
            elif results[3]:
                cached_template, cached_id = results[:2]
                parameter_lengths = results[4]
                print("===========================================")
                print(f"Line-{idx}/{total_line}: ambiguous cache match. {log}")
                print("cached_template: ", cached_template)
                new_template = self.query_ambiguity_candidate(log, regs_common)
                cache.mark_ambiguity_reviewed(cached_id, parameter_lengths)

                if (
                    new_template is not None
                    and cache.is_more_specific(new_template, cached_template)
                ):
                    template_id = cache.add_templates(
                        new_template,
                        source_log=log,
                    )
                    print("accepted_specific_template: ", new_template)
                else:
                    template_id = cached_id
                    print("keeping_cached_template: ", cached_template)

                log_messages.append((log, idx))
                log_templates.append((template_id, idx))
                print("===========================================")
                return True
            else:
                log_messages.append((log, idx))
                log_templates.append((results[1], idx))
                return False


    def load_data(self):
        csv_path = os.path.join(self.path, self.logName+'_structured.csv')
        if os.path.exists(csv_path):
            self.df_log = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        else:
            headers, regex = self.generate_logformat_regex(self.log_format)
            self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)


    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe 
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf


    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
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
