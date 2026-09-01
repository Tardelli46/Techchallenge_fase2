# submissao/

O PDF de submissão reúne a identificação do grupo e os links da entrega. Ele não é
gerado à mão: os dados são preenchidos no JSON e o script monta o documento, mantendo
o formato padronizado.

## Passo a passo

```bash
pip install reportlab
# 1. edite submissao/entrega.json
python submissao/gerar_submissao.py
```

Saída: `submissao/submissao_<TURMA>_<GRUPO>.pdf`

O script **recusa** gerar o PDF se algum valor de exemplo continuar no JSON, se um RM
estiver fora do formato `RM000000` ou se algum link não começar com `https://`.

## Sobre os links

| Link | Requisito |
|---|---|
| Repositório | público, para que qualquer pessoa com o link consiga acessar |
| Apresentação | PDF. Pode apontar para o arquivo em `docs/` do próprio repositório |

Teste os dois em uma **janela anônima** antes de enviar. É o erro mais comum da entrega:
o link funciona na máquina de quem criou e falha para quem abre depois.
