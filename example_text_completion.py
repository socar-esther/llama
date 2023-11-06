import os
import ast
import fire

import pandas as pd
import numpy as np

from llama import Llama
from typing import List

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 100,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # prompts: List[str] = list()
    batch_prompts: List[str] = list()
    llama_output: List[str] = list()
    
    df = pd.read_csv('./dataset/task4_all.csv') # input
    
    for idx in range(len(df)):
        if idx % 10 == 0:
            print(f'> check idx: {idx}/{len(df)}')
            
        row = df.iloc[idx]
        new_cnslt_utter = row['cnslt_conversation']
        new_user_utter = row['user_conversation']
        
        ## TODO: 이부분 처리
        tmp_dict = row['slot_filling_gt']
        tmp_dict = ast.literal_eval(tmp_dict)
        given_ontology = dict()
        for key, value in tmp_dict.items():
            given_ontology[key] = None
        
        if (idx%max_batch_size == 0) & (idx !=0):
            results = generator.text_completion(
                batch_prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            
            for prompt, result in zip(batch_prompts, results):
                print(prompt)
                print(f"> {result['generation']}")
                print("\n==================================\n")
                
                llama_output.append(result['generation'])
                tmp_output = pd.DataFrame({
                    'generated':llama_output})
                tmp_output.to_csv('output/task4_zero_result.csv', index=False)
                
            batch_prompts = list()

        llama_prompt = """
    You can fill in the null values in [Ontology] by referencing [Input]. If you cannot fill in a value from [Input] it is represented as null. In this context, [Output] must always be maintained in JSON format.
    
    [Input]
    상담사: %s
    고객: %s
    [Ontology] %s => """ % (new_cnslt_utter, new_user_utter, given_ontology)
        
        batch_prompts.append(llama_prompt)
    
    
    df['result'] = llama_output
    df.to_csv('output/task4_zero_result.csv', index=False)



if __name__ == "__main__":
    fire.Fire(main)