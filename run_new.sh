#!/bin/bash
#!/bin/bash
#SBATCH --job-name=semibin
#SBATCH --output=outERR14206411.txt
#SBATCH --error=errERR14206411.txt
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:1

# python -m SemiBin.main single_easy_bin \
#     --input-fasta /home/wangjingyuan/lyf/semibin_data_try/7023/contigs.fasta \
#     --input-bam /home/wangjingyuan/lyf/semibin_data_try/7023/ERR14207023_contig.mapped.sorted.bam \
#     --output /home/wangjingyuan/lyf/semibin_data_try/7023/changshi_new1



# python -m SemiBin.main single_easy_bin \
#     --input-fasta /home/wangjingyuan/lyf/data_mar/fastq_data/ERR594296/ERR594296_megahit_out/final.contigs.fa \
#     --input-bam /home/wangjingyuan/lyf/data_mar/fastq_data/ERR594296/ERR594296.sorted.bam \
#     --output /home/wangjingyuan/lyf/data_mar/fastq_data/ERR594296/my_semibin2_output

python -m SemiBin.main single_easy_bin \
    --input-fasta /home/wangjingyuan/lyf/data_mar/fastq_data/ERR594298/ERR594298_megahit_out/final.contigs.fa \
    --input-bam /home/wangjingyuan/lyf/data_mar/fastq_data/ERR594298/ERR594298.sorted.bam \
    --output /home/wangjingyuan/lyf/data_mar/fastq_data/ERR594298/my_semibin2_output