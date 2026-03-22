import ollama
import json
import os
from lxml import etree

def extrair_triplas_com_llama(texto):
    """Envia o texto para o Llama-3 e pede o Grafo em JSON."""
    
    prompt = f"""
    Você é um especialista em Processamento de Linguagem Natural Médica.
    Analise o prontuário abaixo e extraia entidades e relações no formato de triplas (Sujeito, Relação, Objeto).
    
    Regras:
    1. Retorne APENAS um JSON puro, sem explicações.
    2. Use o formato: {{"triplas": [{{"s": "entidade1", "r": "RELACAO", "o": "entidade2"}}]}}
    3. Foque em: Diagnósticos, Sintomas, Medicamentos e Estados do Paciente.
    
    Texto: {texto[:1000]}  # Limitamos para não estourar o contexto inicial
    """

    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': prompt},
    ])
    
    return response['message']['content']

# Teste rápido com o primeiro arquivo
if __name__ == "__main__":
    # Supondo que você tem o texto do arquivo 8908.xml do teste anterior
    exemplo_texto = "paciente lúcida, chorosa, com edema em região craniana, hemiparesia à direita"
    
    print("Enviando para o Llama-3...")
    resultado = extrair_triplas_com_llama(exemplo_texto)
    print("\n--- Resultado da IA ---")
    print(resultado)