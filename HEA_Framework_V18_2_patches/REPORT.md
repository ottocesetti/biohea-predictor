# HEA Framework V18.2 — Patch Report

## Implementado

1. **Simplex tolerante a dados experimentais reais**
   - Fecha composições com ruído de arredondamento.
   - Rejeita negativos e não-finitos.
   - Não exige `sum(x_i) == 1` com tolerância irrealista.

2. **Multiplicative replacement restaurado**
   - ILR volta a receber composições estritamente positivas.
   - Zeros não geram `log(0)` nem `-inf`.

3. **Convenção termodinâmica única**
   - `GM`, `HM`, `SM` são agora canônicos no engine e no descriptor.
   - Remove incompatibilidade silenciosa `gm/hm/sm` vs `GM/HM/SM`.

4. **Cache termodinâmico seguro**
   - Remove cache baseado em `len(X)`.
   - Usa fingerprint da tabela composicional.

5. **Falha termodinâmica taxonomizada**
   - Diferencia:
     - timeout;
     - convergência/divergência;
     - equilíbrio inviável;
     - licença/importação;
     - falha desconhecida.

6. **Missingness termodinâmico explícito**
   - Adiciona:
     - `GM_missing_flag`;
     - `HM_missing_flag`;
     - `SM_missing_flag`.
   - A imputação não fica invisível para o modelo.

7. **Applicability Domain reforçado**
   - Mantém raw score.
   - Adiciona percentil empírico.
   - Adiciona EVT tail-risk.
   - Preserva geometrias separadas.

8. **Diagnóstico de latent space**
   - trustworthiness;
   - continuidade por vizinhança;
   - estabilidade bootstrap;
   - correlação de distâncias entre embeddings.

9. **UMAP auditável**
   - Método alternativo `PCA`.
   - Fallback para PCA se UMAP não estiver instalado.
   - Clipping opcional com warning.

10. **Diagnósticos adicionais**
    - AD dominance;
    - termo-missingness;
    - screening conservador de interações simbólicas;
    - feature diagnostics com VIF, PCA collapse, MI, SHAP opcional e permutation.

## Melhorias que adicionei além do pedido

- `latent_method="pca"` como baseline obrigatório para ablation.
- Redução automática de `n_neighbors` e `n_components` em datasets pequenos.
- `causal_claim=False` nas interações para evitar overclaiming.
- `combined_evt_tail_risk` separado do `combined_percentile_risk`.
- `out_of_domain_flag` baseado em percentil, não em distância bruta.

## Limitações restantes

1. UMAP continua não físico e não causal.
2. EVT precisa de cauda suficiente para ser confiável.
3. Mahalanobis ainda representa um regime elipsoidal único.
4. Imputação termodinâmica ainda não propaga incerteza completa.
5. Interações simbólicas precisam de nested CV antes de entrarem no modelo principal.
