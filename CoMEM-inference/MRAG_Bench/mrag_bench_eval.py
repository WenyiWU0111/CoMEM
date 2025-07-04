import os 
import json 
import argparse
from collections import OrderedDict
from utils.gpt_extract import extract_answer
from tqdm.auto import tqdm

if __name__ == "__main__":
   
    input_file = ""

    if input_file.split('.')[-1] == 'json':
        with open(input_file, 'r') as f:
            res = json.load(f)
    else:
        res = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                res.append(json.loads(line))
    
    outputpath = input_file.split('.')[0].split('/')[-1] + '_score.json'
    if not os.path.exists('CoMEM-inference/MRAG_Bench/result/clip/eval'):
        os.makedirs("CoMEM-inference/MRAG_Bench/result/clip/eval")
    outputpath = os.path.join('CoMEM-inference/MRAG_Bench/result/clip/eval', outputpath)
    outfile = {}

    need_extra_extract = []
    gt = []
    pred = []
    cat = []
    for item in tqdm(res):

        gt.append(item['gt_choice'])
        out = item['output'].strip()[0]
        
        if out not in ['A', 'B', 'C', 'D']:
            # print(out)
            problem = item['prompt']
            extraction  = extract_answer(item['output'], problem)
            out = extraction
            item['extracted_output'] = extraction
            need_extra_extract.append(item)
            if out not in ['A', 'B', 'C', 'D']:
                print('Error, extraction failed for this response:', item['output'])
        print('output',out)
        pred.append(out)
        cat.append(item['scenario'])   
            
    acc = 0
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            acc += 1
    overall_accuracy = acc / len(gt) * 100
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    outfile['overall_accuracy'] = round(overall_accuracy, 2)
    print('='*50)
    all_scenarios = set([item['scenario'] for item in res if 'scenario' in item])

    for scenario in all_scenarios:

        scenario_correct = {}
        scenario_total = {}

        for i in range(len(gt)):

            cur_scenariol = cat[i]
            
            if cur_scenariol != scenario:
                continue
            
            if scenario not in scenario_total:
                scenario_total[scenario] = 0
                scenario_correct[scenario] = 0
            
            scenario_total[scenario] += 1
            if gt[i] == pred[i]:
                scenario_correct[scenario] += 1
        
        scenario_overall = sum(scenario_correct.values()) / sum(scenario_total.values()) * 100 
        scenario_overall =  round(scenario_overall, 2)
        print(scenario + ": ", scenario_overall)
        print(scenario + " total: ", sum(scenario_total.values()))
        outfile[scenario] = scenario_overall
        #print('-----------------------------------------------------------')

    key_order = [
        'Overall',
        'Angle',
        'Partial',
        'Scope',
        'Occlusion',
        'Temporal',
        'Deformation',
        'Incomplete',
        'Biological',
        'Others',
    ]

    ordered_outputfile = OrderedDict((key, outfile[key]) for key in key_order if key in outfile)
    ordered_outputfile = dict(ordered_outputfile)
    outfile = ordered_outputfile

    with open(outputpath, 'w') as f:
        json.dump(outfile, f, indent=4)

    extra_output_path = outputpath.split('.')[0] + '_gpt_extracted.json'
    if len (need_extra_extract) > 0: 
        with open(extra_output_path, 'w') as f:
            json.dump(need_extra_extract, f, indent=4)
    