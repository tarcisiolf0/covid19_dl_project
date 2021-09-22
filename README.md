# Projeto COVID-19
## Implementação de Rede Neural Convolucional para classificação de imagens de Raio-X de tórax como: COVID, Pneumonia ou Normal. 

### Dupla: Carlos e Tarcísio
#### Período Letivo 2020.2

Apresentação: https://youtu.be/8UGl7ZgCr2Y

## Artigo
O projeto foi baseado no artigo: *COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest X-Ray Images*.
O *papper* está disponível em <https://paperswithcode.com/paper/covid-net-a-tailored-deep-convolutional>. 
Já o repositório original de implementação da rede está disponível em: <https://github.com/lindawangg/COVID-Net>.

## Proposta do Trabalho
O proposta do projeto é treinar o modelo proposto no artigo em uma nova base de dados. O dataset escolhido foi o dataset de competição do Kaggle "*SIIM-FISABIO-RSNA COVID-19 Detection*",
disponível em <https://www.kaggle.com/c/siim-covid19-detection/data>.

#### Requisitos para rodar:
Os requisitos principais são:

- Tensorflow 1.13 e 1.15
- OpenCV 4.2.0
- Python 3.6
- Numpy
- Scikit-Learn
- Matplotlib

Requisitos adicionais para gerar o *dataset*:

- PyDicom
- Pandas
- Jupyter

### Executando
Para maiores detalhes de como instalar os requisitos necessários, para gerar o dataset, 
para fazer o treino, validação e inferência da COVIDNet, basta ler o README do repositório <https://github.com/lindawangg/COVID-Net>.

O projeto foi desenvolvido utilizando o Spyder e o Jupyter Notebook.
