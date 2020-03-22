# wine_quality

Repositório responsável pela avaliação da qualidade de vinhos portugueses. São 12 características dos vinho que serão avaliadas para discriminar a qualidade do mesmo. São estes os atributos:

* 0   type                  object 
* 1   fixed acidity         float64
* 2   volatile acidity      float64
* 3   citric acid           float64
* 4   residual sugar        float64
* 5   chlorides             float64
* 6   free sulfur dioxide   float64
* 7   total sulfur dioxide  float64
* 8   density               float64
* 9   pH                    float64
* 10  sulphates             float64
* 11  alcohol               object 
* 12  quality               int64

Vamos realizar uma análise de dados e responder algumas perguntas que serão dispostas a seguir:

## a. Como foi a definição da sua estratégia de modelagem?

Para cada atributo, verifiquei sua influência para com a qualidade. Neste sentido, a partir da observação gráfica dos dados e seus padrões comportamentais, poderiamos excluir sua participação nos treinos do modelo, se para cada índice de qualidade não houver variação correlativa.
Observado alguma correlação, seja de forma crescente ou decrescente, podemos ver alguma correlação e manter o atributo, podendo este se derivar em outros novos atributos (lógica fuzzy), ou defragmentar-se.

No decorrer da modelagem são explicadas, em comentários de código, as ações realizadas para modelagem dos dados para cada atributo.

## b. Como foi definida a função de custo utilizada?

Para a abordagem de deep learning, treinamos uma rede neural com a loss function categorical_crossentropy. Isso porque nossos rótulos, apesar de serem números, podem ser representados por categorias de 3 à 9, onde cada vinho é rodulado apenas 1 vez. Desta forma, a loss function nos dá a diferença entre a matriz softmax resultante da matriz softmax ideal. Essa matriz ideal é uma one hot encoding, já a resultante é uma matriz de probabilidades preditas.

Como bônus, foi implementado a função MSE que poderia ser vista como uma função de custo para todas as abordagens, ela é interessante porque amplia os erros dos modelos de forma que o erro pode apresentar uma gravidade maior dependendo da distância da resposta correta.

## c. Qual foi o critério utilizado na seleção do modelo final?

Para seleção deste modelo específico, de uma forma rápida, escolheria o com maior acurácia, que no nosso caso foi RandomForestClassifier com aproximadamente 67%. Porém, para análise mais profunda, podemos avaliar o quão grave o modelo erra, quando erra. Há casos que a gravidade do erro pode comprometer todo o interesse de usar o modelo. Nestes casos, gosto de avaliar com a métrica RMSE, para verificar o quão distante da resposta certa estão as predições do modelo.

## d. Qual foi o critério utilizado para validação do modelo? Por que escolheu utilizar este método?

A validação de um modelo é uma etapa um pouco complexa. Esses modelos não foram validados. Para uma validação digna, necessitaria chamar uma pessoa da área, que seria um especialista em vinhos, e verificar o quão bom o modelo responde ao que esse especialista responde.

## e. Quais evidências você possui de que seu modelo é suficientemente bom?

A partir das observações, as maiores evidências que o modelo é bom são seus resultados em termos de métricas (aqui vem boas escolhas delas, seja acurácia, precisão, f-measure, informedness, markedness). Para esse exemplo da qualidade do vinho, escolhemos acurácia e rmse para verificarmos o modelo, e a partir de um pequeno "benchmarking", encontramos a melhor ténica que poderia ser utilizada para realizar as predições de qualidade.

# Dicas de uso:

* 1 - Crie um ambiente virtual em python
* 2 - Instale os pacotes nescessários com pip install -r requiremnets.txt
* 3 - Execute cada arquivo .py isoladamente ou com extensão jupyter notebook do visualcode parcialmente
