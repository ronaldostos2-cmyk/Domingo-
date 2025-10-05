# Bot Trader Binance Testnet (Transformer Auto-Learning)

Projeto modular com IA avançada (Transformer autoaprendente) para operar na Binance Testnet.

## Como usar
1. Copie `.env.example` para `.env` e preencha suas chaves do Testnet (https://testnet.binance.vision/).
2. Crie um virtualenv e instale dependências:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Para rodar o monitor (Streamlit):
   ```bash
   streamlit run src/monitor.py
   ```
4. Para rodar o bot:
   ```bash
   python -m src.main
   ```

## Notas
- Este projeto foi gerado automaticamente e serve como base. Teste no **Testnet** antes de qualquer mudança para Mainnet.
- O modelo usa um Transformer encoder leve com dropout MC para estimativa de incerteza e experience replay para aprendizado online.
