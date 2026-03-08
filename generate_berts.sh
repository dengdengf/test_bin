#!/bin/bash
#SBATCH --job-name=fur
#SBATCH --output=fur_output1.txt
#SBATCH --error=fur_error1.txt
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:1

python /home/wangjingyuan/lyf/SemiBin-main/SemiBin/generate_berts.py -md /home/wangjingyuan/lyf/DCVBin_project/DNABERT-S -fd /home/wangjingyuan/lyf/semibin_data_try/7023/contigs.fasta -sd /home/wangjingyuan/lyf/semibin_data_try/7023/tuhuan1/seq.txt -cf /home/wangjingyuan/lyf/semibin_data_try/7023/tuhuan1/contig_name.txt -dd /home/wangjingyuan/lyf/semibin_data_try/7023/tuhuan1 --csv_file /hom
e/wangjingyuan/lyf/semibin_data_try/7023/tuhuan1


# python -m SemiBin.main     single_easy_bin     --input-fasta /home/wangjingyuan/lyf/semibin_data_try/7023/contigs.fasta     --input-bam /home/wangjingyuan/lyf/semibin_data_try/7023/ERR14207023_contig.mapped.sorted.bam     --output /home/wangjingyuan/lyf/semibin_data_try/7023/tuhuan1