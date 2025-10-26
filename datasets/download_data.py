# DOWNLOAD_DATA.py
import os

# 1. Instalar o pacote oficial do Kaggle
os.system("pip install kaggle")

# 2. Baixar o dataset
# Certifique-se de ter o token Kaggle API configurado (~/.kaggle/kaggle.json)
os.system("kaggle datasets download -d yuricordeiro/tcc-mba-ia-usp -p data")

# 3. Extrair o conteúdo
os.system("unzip data/tcc-mba-ia-usp.zip -d data")
