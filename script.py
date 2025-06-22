from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, GPT2Tokenizer
from typing import List
from collections import defaultdict
import torch

app = FastAPI(title="API de Análise de Emoções com BERT-pt")

# Configuração dos modelos
device = 0 if torch.cuda.is_available() else -1

models = {
    "sentimento": pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=device
    ),
    "resumo": pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=device
    ),
    "topico": pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device=device
    ),
    "keywords": pipeline(
        "token-classification",
        model="Davlan/bert-base-multilingual-cased-ner-hrl",
        device=device
    ),
    "emocoes": pipeline(
        "text-classification",
        model="AnasAlokla/multilingual_go_emotions",
        device=device,
        return_all_scores=False,
        top_k=3
        #top_k=None  # Para retornar todas as emoções
    )
}

# Mapeamento de emoções para português
EMOTION_MAP = {
    "anger": "raiva",
    "disgust": "nojo",
    "fear": "medo",
    "joy": "alegria",
    "neutral": None,
    "others": None,
    "sadness": "tristeza",
    "surprise": "surpresa",
    "annoyance": "irritação",
    "disappointment": "decepção",
    "gratitude": "gratidão",
    "confusion": "confusão",
    "frustration": "frustração"
}

class Texto(BaseModel):
    texto: str

class TextoComCategorias(BaseModel):
    texto: str
    categorias: List[str]

class Mensagem(BaseModel):
    remetente: str
    texto: str

class Categoria(BaseModel):
    id: int
    categoria: str

class HistoricoTicket(BaseModel):
    historico: List[Mensagem]
    categorias: List[Categoria]  # Lista de objetos Categoria

class Historico(BaseModel):
    historico: List[Mensagem]

def traduzir_emocao(emocao: str) -> str:
    return EMOTION_MAP.get(emocao.lower(), emocao)

def classificar_sentimento(mensagem: str) -> str:
    resultado = models["sentimento"](mensagem)[0]
    estrelas = int(resultado['label'].split()[0])
    return "positivo" if estrelas >= 4 else "neutro" if estrelas == 3 else "negativo"


def pos_processar_emocoes(emocoes: list) -> list:
    # Filtra neutro quando há outras emoções relevantes
    if len(emocoes) > 1 and emocoes[0]["emocao"] == "neutro":
        # Se a diferença para a segunda emoção for pequena, descarta neutro
        if emocoes[0]["score"] - emocoes[1]["score"] < 0.15:
            return emocoes[1:]
    return emocoes


def analisar_emocoes(texto: str) -> tuple:
    try:
        resultado = models["emocoes"](texto)
        emocoes = []

        scores = [item['score'] for item in resultado[0]]
        max_score = max(scores) if scores else 0
        threshold = max(0.1, max_score * 0.3)  # Pelo menos 0.1, ou 30% do máximo

        for item in resultado[0]:
            emocao_traduzida = traduzir_emocao(item['label'])
            if item['score'] >= threshold:
                emocoes.append({
                    "emocao": emocao_traduzida,
                    "score": round(item['score'], 4)
                })

        if not emocoes:
            return {"emocao": "neutro", "score": 1.0}, []

        emocoes.sort(key=lambda x: x['score'], reverse=True)

        # Só aceita neutro se for claramente dominante (score > 0.6)
        if emocoes[0]["emocao"] == "neutro" and emocoes[0]["score"] < 0.6:
            if len(emocoes) > 1:
                return emocoes[1], emocoes[:3]
            return {"emocao": "neutro", "score": 1.0}, []

        return emocoes[0], emocoes[:3]
    except Exception as e:
        return {"emocao": "neutro", "score": 1.0}, []

@app.post("/analisar")
async def analisar_sentimento(texto: Texto):
    sentimento = classificar_sentimento(texto.texto)
    emocao_principal, _ = analisar_emocoes(texto.texto)
    return {
        "texto": texto.texto,
        "sentimento": sentimento,
        "emocao_principal": emocao_principal
    }

@app.post("/resumir")
async def resumir_conversa(historico: List[Mensagem], limite_tokens: int = 100):
    mensagens_cliente = [m.texto for m in historico if m.remetente.lower() == "cliente"]
    if not mensagens_cliente:
        return {"erro": "Nenhuma mensagem do cliente encontrada"}

    texto = " ".join(mensagens_cliente)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(texto)

    if len(tokens) > limite_tokens:
        tokens = tokens[-limite_tokens:]
        texto = tokenizer.decode(tokens)

    try:
        resumo = models["resumo"](texto, max_length=150, min_length=30)[0]["summary_text"]
        return {"resumo": resumo}
    except Exception as e:
        return {"erro": str(e)}

@app.post("/classificar")
async def classificar_topico(data: TextoComCategorias):
    resultado = models["topico"](data.texto, candidate_labels=data.categorias)
    classificacoes = []
    for label, score in zip(resultado["labels"], resultado["scores"]):
        classificacoes.append({
            "categoria": label,
            "pontuacao": round(score, 3),
            "relevancia": "alta" if score >= 0.7 else "média" if score >= 0.4 else "baixa"
        })
    return {
        "texto": data.texto,
        "classificacoes": classificacoes
    }

@app.post("/extrair-palavras-chave")
async def extrair_keywords(texto: Texto):
    entidades = models["keywords"](texto.texto)
    palavras_chave = []
    palavra_atual = ""
    tipo_entidade = ""

    for ent in entidades:
        if ent["entity"].startswith("B-"):
            if palavra_atual:
                palavras_chave.append({"termo": palavra_atual.strip(), "tipo": tipo_entidade})
            palavra_atual = ent["word"].replace("##", "")
            tipo_entidade = ent["entity"][2:]
        elif ent["entity"].startswith("I-") and tipo_entidade:
            palavra_atual += " " + ent["word"].replace("##", "")

    if palavra_atual:
        palavras_chave.append({"termo": palavra_atual.strip(), "tipo": tipo_entidade})

    return {
        "texto": texto.texto,
        "palavras_chave": palavras_chave
    }

@app.post("/categorizar-ticket")
@app.post("/categorizar-ticket")
async def categorizar_ticket(dados: HistoricoTicket):
    texto_cliente = "\n".join([m.texto for m in dados.historico if m.remetente.lower() == "cliente"])

    # Extrair apenas os nomes das categorias
    categorias_nomes = [cat.categoria for cat in dados.categorias]
    resultado = models["topico"](texto_cliente, candidate_labels=categorias_nomes)

    classificacao_termos = []
    for label, score in zip(resultado["labels"], resultado["scores"]):
        categoria_id = next((cat.id for cat in dados.categorias if cat.categoria == label), None)

        classificacao_termos.append({
            "id": categoria_id,  # ID correto
            "categoria": label,  # Nome da categoria
            "pontuacao": round(score, 2),
            "classificacao": "Pouco relevante" if score < 0.5 else "Relevante" if score < 0.75 else "Muito relevante"
        })

    return {
        "texto_formatado": texto_cliente,
        "classificacao": classificacao_termos
    }


@app.post("/analise-generica")
async def analise_generica(historico: Historico):
    try:
        mensagens_cliente = [m for m in historico.historico if m.remetente.lower() == "cliente"]
        if not mensagens_cliente:
            return {"erro": "Nenhuma mensagem do cliente encontrada"}

        resultados = {
            "total_mensagens": len(mensagens_cliente),
            "sentimentos": defaultdict(int),
            "emocoes_principais": defaultdict(int),
            "top_emocoes": defaultdict(float),
            "emocoes_por_mensagem": []
        }

        for msg in mensagens_cliente:
            sentimento = classificar_sentimento(msg.texto)
            emocao_principal, top_emocoes = analisar_emocoes(msg.texto)

            resultados["sentimentos"][sentimento] += 1
            resultados["emocoes_principais"][emocao_principal["emocao"]] += 1
            for emocao in top_emocoes:
                resultados["top_emocoes"][emocao["emocao"]] += emocao["score"]
            resultados["emocoes_por_mensagem"].append({
                "texto": msg.texto,
                "sentimento": sentimento,
                "emocao_principal": emocao_principal,
                "top_emocoes": top_emocoes
            })

        # Filtra emoções neutras do cálculo da emoção mais frequente
        emocoes_freq = {
            k: v for k, v in resultados["emocoes_principais"].items()
            if k != "neutro"
        }

        # Se todas foram neutras, mantém o neutro
        if not emocoes_freq:
            emocoes_freq = {"neutro": resultados["emocoes_principais"]["neutro"]}

        emocao_frequente = max(
            emocoes_freq.items(),
            key=lambda x: x[1],
            default=("neutro", 0)
        )[0]

        # Calcula top emoções (média de scores)
        top_3_emocoes = sorted(
            [(emocao, score / len(mensagens_cliente))
             for emocao, score in resultados["top_emocoes"].items()],
            key=lambda x: x[1],
            reverse=True
        )

        # Filtra neutro com score baixo
        top_3_emocoes = [e for e in top_3_emocoes if e[0] != "neutro" or e[1] > 0.5][:3]

        return {
            "analise_geral": {
                "sentimento_geral": "positivo" if resultados["sentimentos"]["positivo"] > resultados["sentimentos"][
                    "negativo"]
                else "negativo" if resultados["sentimentos"]["negativo"] > resultados["sentimentos"]["positivo"]
                else "neutro",
                "distribuicao_sentimentos": dict(resultados["sentimentos"]),
                "emocao_mais_frequente": emocao_frequente,
                "top_emocoes": [{"emocao": e[0], "score_medio": round(e[1], 4)} for e in top_3_emocoes],
                "total_mensagens": len(mensagens_cliente)
            },
            "analise_detalhada": resultados["emocoes_por_mensagem"]
        }

    except Exception as e:
        return {"erro": str(e)}

@app.post("/analise-emocoes-detalhada")
async def analise_emocoes_detalhada(texto: Texto):
    try:
        resultado = models["emocoes"](texto.texto)
        emocoes = []

        for item in resultado[0]:
            emocao = traduzir_emocao(item['label'])
            if item['score'] > 0.1:
                emocoes.append({
                    "emocao": emocao,
                    "score": round(item['score'], 4),
                    "status": "detectado"
                })

        texto_lower = texto.texto.lower()
        if "obrigado" in texto_lower:
            emocoes.append({
                "emocao": "gratidão",
                "score": 0.85,
                "status": "contextual"
            })
        if "frustrado" in texto_lower:
            emocoes.append({
                "emocao": "frustração",
                "score": 0.9,
                "status": "contextual"
            })

        emocoes.sort(key=lambda x: x['score'], reverse=True)

        return {
            "texto": texto.texto,
            "emocao_principal": emocoes[0] if emocoes else None,
            "todas_emocoes": emocoes[:5]
        }

    except Exception as e:
        return {"erro": str(e)}

@app.get("/status")
async def status():
    return {
        "status": "ativo",
        "modelos": list(models.keys()),
        "dispositivo": "GPU" if torch.cuda.is_available() else "CPU",
        "modelo_emocoes": "pucpr/bert-base-portuguese-cased-emotion"
    }
