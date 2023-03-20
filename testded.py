import pandas as pd
import matplotlib.pyplot as ply
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

tabela = pd.read_csv('basedados1.csv')
print(tabela)
print(tabela.info())
print(tabela.corr())
sn.heatmap(tabela.corr(), annot=True, cmap="Wistia")
ply.show()
y = tabela["Vendas"]
#y é quem eu quero prever
#x é quem eu vou usar pra aprender
#encoding é bacana pesquisar
x = tabela.drop("Vendas", axis=1)
x_treino,x_teste,y_treino,y_teste = train_test_split(x, y, test_size= 0.3)

modelo_regressãoLinear = LinearRegression()
modelo_arvoreDecisão = RandomForestRegressor()

modelo_regressãoLinear.fit(x_treino, y_treino)
modelo_arvoreDecisão.fit(x_treino, y_treino)
RandomForestRegressor()

previsão_regressãolinear = modelo_regressãoLinear.predict(x_teste)
previsão_arvoredecisão = modelo_arvoreDecisão.predict(x_teste)
print(r2_score(y_teste, previsão_regressãolinear))
print(r2_score(y_teste, previsão_arvoredecisão))

tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsoes ArvoreDecisao"] = previsão_arvoredecisão
tabela_auxiliar["Previsoes Regressao Linear"] = previsão_regressãolinear

sn.lineplot(data=tabela_auxiliar)
ply.show() 
