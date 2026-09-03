# VRPTW via Relaxação Lagrangeana, Branch-and-Bound e Branch-and-Cut

Projeto final (COS890) para resolver o **Vehicle Routing Problem with Time
Windows (VRPTW)**, no formato de instâncias de **Solomon**, por três métodos
de otimização combinatória, todos em Python e apoiados no **Gurobi**
(`gurobipy`) sempre que possível.

## Objetivo e convenções

Segue o objetivo lexicográfico clássico do benchmark de Solomon:

1. minimizar o número de veículos (rotas usadas);
2. entre as soluções com o número mínimo de veículos, minimizar a distância
   total percorrida.

Isso é resolvido em **duas fases** (ver `src/vrptw/solvers.py`): a fase 1
minimiza o número de veículos, a fase 2 fixa esse número e minimiza a
distância.

## Formulação matemática (comum ao B&B e ao B&C)

Modelo arc-flow 2-index compacto (`src/vrptw/formulation.py`), nó `0` é o
depósito, nós `1..n` são clientes:

- `x[i,j] ∈ {0,1}`: arco `(i,j)` usado;
- `w[i] ∈ [e_i, l_i]`: instante de início de atendimento no cliente `i`;
- `u[i] ∈ [d_i, Q]`: carga acumulada até e incluindo o cliente `i`.

Restrições principais:

- grau de saída/entrada = 1 para cada cliente (visita exatamente uma vez);
- grau de saída do depósito ≤ (ou =) número de veículos, grau de
  saída = grau de entrada no depósito;
- **linkagem de tempo** (também elimina subciclos, já que os tempos de
  viagem são estritamente positivos):
  `w[j] ≥ w[i] + s_i + t_ij − M_ij·(1 − x[i,j])`;
- **linkagem de carga** (estilo MTZ): `u[j] ≥ u[i] + d_j − Q·(1 − x[i,j])`.

Como a linkagem de tempo já impede subciclos que não passam pelo depósito,
**nenhuma restrição de eliminação de subciclo (SEC) é necessária para a
corretude do modelo** — o modelo compacto já é 100% correto sozinho. Os
cortes adicionados no Branch-and-Cut servem só para **fortalecer o bound de
LP**, não para garantir viabilidade.

## Os três métodos

| Método | Arquivo | Como funciona |
|---|---|---|
| Branch-and-Bound | `solvers.py::solve_branch_and_bound` | Resolve o modelo compacto com o Gurobi, mas com `Cuts=0` e `Heuristics=0`: a árvore de busca é guiada só pelo bound de LP relaxado e pelo branching, sem os cortes automáticos do Gurobi. |
| Branch-and-Cut | `solvers.py::solve_branch_and_cut` | Mesmo modelo, mas com os cortes automáticos do Gurobi ligados **e** um callback que separa, em nós fracionários, cortes de **capacidade arredondada (RCC)** e **infeasible-path cuts** (`cuts.py`). |
| Relaxação Lagrangeana | `lagrangian.py::lagrangian_relaxation` | Dualiza a restrição "cliente visitado exatamente uma vez"; o subproblema decomposto é um caminho mínimo elementar com restrição de recursos (**ESPPRC**: janela de tempo + capacidade), resolvido por *label-setting* próprio (sem Gurobi, pois é um problema combinatório específico melhor resolvido por DP). O número mínimo de veículos usado como `K` no subproblema é estimado rapidamente com o Gurobi. Atualização dos multiplicadores por subgradiente (regra de Polyak/Held-Karp). |

### Limitações conhecidas (leia antes de rodar em instâncias grandes)

- **Licença do Gurobi**: sem uma licença acadêmica configurada, `gurobipy`
  usa a licença gratuita *size-limited* (≤ 2000 variáveis/restrições). O
  modelo compacto para uma instância de **25 clientes cabe** (~700
  variáveis), mas **50 clientes não cabe** (~2650 variáveis) e o solve
  falha com `Model too large for size-limited license`. Como você está em
  um programa de pós-graduação, vale pegar uma licença acadêmica gratuita
  (Named-User ou WLS) em gurobi.com/academia e ativá-la com `grbgetkey` —
  depois disso as instâncias de 50 clientes funcionam sem mudar nada no
  código.
- **ESPPRC em Python puro**: o *label-setting* elementar (com dominância por
  bitmask de clientes visitados) é implementado do zero e pode ficar lento
  para instâncias com muitos clientes compatíveis entre si (ex.: as
  instâncias `c1xx`, bem "agrupadas"). Por isso `lagrangian_relaxation` tem
  um `time_limit` e um `max_labels`; se o subproblema for cortado antes de
  provar otimalidade, o resultado tem `subproblem_exact=False` e o
  `lower_bound` reportado **deixa de ser um bound certificado** (é só uma
  aproximação). Para instâncias de 25 clientes isso costuma acontecer com
  `time_limit` baixo; aumente o tempo ou reduza o número de clientes para
  bounds exatos.

## Estrutura do projeto

```
src/vrptw/
  instance.py      # parser do formato Solomon + VRPTWInstance
  solution.py       # Route/Solution, validação de viabilidade, custo lexicográfico
  formulation.py     # modelo MILP compacto (gurobipy)
  cuts.py            # separação de RCC e infeasible-path cuts
  solvers.py         # solve_branch_and_bound / solve_branch_and_cut (two-phase)
  lagrangian.py       # ESPPRC (label-setting) + subgradiente
scripts/
  solve.py            # CLI: resolve uma instância com um dos 3 métodos
  run_benchmark.py     # roda os 3 métodos em várias instâncias, salva CSV
  plot_results.py       # gera gráficos comparativos a partir do CSV
tests/                  # pytest (parser, solução, solvers, Lagrangeana)
data/solomon/raw/        # instâncias Solomon de 100 clientes (.txt)
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Se tiver uma licença Gurobi acadêmica, ative com `grbgetkey` antes de
rodar instâncias com mais de ~25 clientes.

## Uso

Resolver uma instância:

```powershell
python scripts/solve.py data/solomon/raw/c101.txt --method bc --customers 25 --time-limit 120
python scripts/solve.py data/solomon/raw/c101.txt --method bb --customers 25 --time-limit 120
python scripts/solve.py data/solomon/raw/c101.txt --method lr --customers 25 --time-limit 120
```

Rodar um benchmark comparando os 3 métodos em várias instâncias:

```powershell
python scripts/run_benchmark.py --instances c101 c201 r101 r201 rc101 rc201 --customers 25 --time-limit 120
python scripts/plot_results.py results/benchmark_25.csv
```

## Testes

```powershell
pytest -q
```
