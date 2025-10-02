#!/usr/bin/env python3
import sys
sys.path.append('./../../')

import argparse
import yaml
import pandas as pd
from processing.utils.pipeline import CTClassificationPipeline

def main():
    conf_path = 'processing/config/inference.yaml'
    
    # Загрузка конфигурации
    with open(conf_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Инициализация пайплайна
    pipeline = CTClassificationPipeline(config)
    
    # Поиск исследований
    import glob
    studies = glob.glob(f"{config['input_path']}/*")
    
    # Обработка
    results_df = pipeline.batch_process(studies)
    
    # Сохранение результатов
    #results_df.to_excel(config['output_path'], index=False)
    print(f"Results saved to {config['output_path']}")

if __name__ == '__main__':
    main()