PROMPT = {
    "('Spanish', 'Spain')": """Pregunta: {} 
Para esta pregunta, por favor realiza un razonamiento paso a paso para obtener la respuesta final.
Debes responder solo con el número de la opción.
Ten en cuenta que la respuesta final debe estar en el siguiente formato:
Proceso de razonamiento: todos los pasos del pensamiento
Respuesta final: \\boxed{{tu número de opción aquí}}""",

    "('Russian', 'Russia')": """Вопрос: {} 
Пожалуйста, выполните пошаговое рассуждение, чтобы получить окончательный ответ.
Вы должны отвечать только номером варианта.
Обратите внимание, что окончательный ответ должен быть в следующем формате:
Ход рассуждений: все шаги мышления
Окончательный ответ: \\boxed{{ваш номер варианта}}""",

    "('Bulgarian', 'Bulgaria')": """Въпрос: {} 
Моля, направете стъпка по стъпка разсъждение, за да получите крайния отговор.
Трябва да отговорите само с номера на опцията.
Обърнете внимание, че крайният отговор трябва да е във формат:
Процес на разсъждение: всички стъпки на мислене
Окончателен отговор: \\boxed{{вашият номер на опция}}""",

    "('Portuguese', 'Brazil')": """Pergunta: {} 
Para esta pergunta, por favor, realize um raciocínio passo a passo para obter a resposta final.
Você deve responder apenas com o número da opção.
Note que a resposta final deve estar no seguinte formato:
Processo de raciocínio: todos os passos do pensamento
Resposta final: \\boxed{{o número da sua opção aqui}}""",

"('Chinese', 'China')": """问题: {} 
请对这个问题进行逐步推理，以得出最终答案。
你应该只回复选项编号。
请注意最终答案的格式如下：
推理过程: 所有思考步骤
最终答案: \\boxed{{你的选项编号}}""",
}