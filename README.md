# GPTito
A small Transformer

Antes de los Transformers, hubo intentos de craer modelos que lograran trabar en paralelo, los hubo pero aún dependias de la secuencialidad. Tenemos como ejemplo Extended Neural GPU que usaba Redes Neuronales Convolucionales en combinacón con Redes Recurrentes, se aprovechaba de las convoluciones para poder trabajar de forma paralela, pero por el componente recurrente aún se dependía de Secuencias, luego está ByteNet que usaba CNN para el procesamiento paralelo y esó era posible para todo el entrenamiento, el problema era que al momento de hacer inferencia aún dependia de la secuencialidad, porque predecia paso a paso. Al final teniamos la arquitectura ConvS2S que fue la logro la parelización por completo sin tener que depender de recurrencias, pero el problema que tenía era que con este último y los otros dos modelos es que mientras más alejadas estuvieran las dependencias, era más dificil que el modelo aprendiera y fuera preciso.

La atención es un mecanismo que relaciona diferentes posiciones en una secuencia, con el proposíto de calcular una representación de la secuencia.
¿Qué es esta representación?
La representación viene siendo una ponderación que se calcula al momento de que la atención en las relaciones importantes que hay en una secuencia.

## Arquitectura
Gptito se compone de

### InputEmbedding

### PostionalEncoding

### MultiHeadAttention

### Encoder

### Decoder
