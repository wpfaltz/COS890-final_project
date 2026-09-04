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

**Importante:** Branch-and-Bound e Branch-and-Cut são implementações **próprias**
(árvore de busca *best-first* escrita à mão) — o Gurobi é usado só como motor de
LP para resolver a relaxação em cada nó, nunca como o MIP solver. Uma versão
automática do Gurobi (`Cuts=0`/cortes via callback, mas com `model.optimize()`
rodando a árvore) existiu numa iteração anterior do projeto e foi identificada
como um problema: ela não passava de um "Gurobi automático disfarçado". Foi
reescrita; a versão que só chama o Gurobi automático continua existindo, mas
isolada em `reference_gurobi.py`, usada **apenas como baseline de comparação**,
nunca contada como um dos 3 métodos exigidos.

| Método | Arquivo | Como funciona |
|---|---|---|
| Branch-and-Bound | `solvers.py::solve_branch_and_bound` | Árvore *best-first* própria (fila de prioridade por bound de LP). Em cada nó, todas as `x` são contínuas ([0,1]) e o Gurobi resolve só essa LP; a integralidade vem do branching na variável mais fracionária (fixada em 0 num filho, em 1 no outro). Nenhum corte além da formulação base. |
| Branch-and-Cut | `solvers.py::solve_branch_and_cut` | Mesma árvore, mas a cada nó (a partir da raiz) roda um loop de planos de corte que separa **rounded capacity cuts (RCC)** e **infeasible-path cuts** (`cuts.py`) contra a solução fracionária, adicionando-os como restrições globais permanentes antes de decidir o branching. |
| Relaxação Lagrangeana | `lagrangian.py::lagrangian_relaxation` | Dualiza a restrição "cliente visitado exatamente uma vez"; o subproblema decomposto é um caminho mínimo com restrição de recursos (**ESPPRC** com relaxação **ng-route**: janela de tempo + capacidade), resolvido por *label-setting* próprio (sem Gurobi), sem nenhum corte artificial no número de labels explorados — só o `time_limit` (relógio) da busca por subgradiente encerra o subproblema. O número mínimo de veículos usado como `K` no subproblema é estimado rapidamente com o Gurobi, que também resolve (uma única vez, sem branching) a relaxação linear do modelo compacto com essa frota fixa para reportar `root_lp_bound` — comparável ao `root_lp_bound` do B&B/B&C. Atualização dos multiplicadores por subgradiente (regra de Polyak/Held-Karp). **Só produz bounds** (`lower_bound` dual + `root_lp_bound` + heurístico de `upper_bound`) — não calcula/reporta uma "distance" própria e não é um método exato. |
| *Baseline (não é um dos 3 métodos)* | `reference_gurobi.py::solve_gurobi_reference` | Entrega o MILP inteiro para o solver MIP nativo do Gurobi (cortes, presolve e heurísticas automáticos). Serve só para validar corretude e comparar eficiência. |

### Limitações conhecidas (leia antes de rodar em instâncias grandes)

- **Licença do Gurobi**: já está configurada uma licença **acadêmica (WLS)**
  nesta máquina (sem o limite de 2000 variáveis/restrições da licença
  gratuita *size-limited*), então instâncias de **50 clientes** (~2650
  variáveis) resolvem normalmente. Se rodar em outra máquina sem licença
  acadêmica configurada, o solve de instâncias grandes pode falhar com
  `Model too large for size-limited license` — nesse caso, pegue uma
  licença acadêmica gratuita (Named-User ou WLS) em gurobi.com/academia e
  ative com `grbgetkey`.
- **B&B/B&C manuais em 100 clientes**: a fase 1 (tamanho da frota) agora é
  resolvida à parte, diretamente pelo solver MIP do Gurobi (`_find_min_vehicles`
  em `solvers.py`) — contar arcos de saída do depósito não tem estrutura de
  roteamento interessante para a árvore manual explorar, e era o gargalo real
  (antes disso, `r101`/`c201` com 100 clientes nem terminavam a fase 1 em 300s).
  Com essa mudança, `r101`/`c201`/100 resolvem em segundos a poucos minutos. As
  famílias `r2xx`/`rc2xx` (horizonte longo, janelas de tempo largas) continuam
  genuinamente difíceis para a árvore manual em 100 clientes — é um limite
  conhecido da literatura de VRPTW exato, não um bug: `r201`/100 não encontra
  nem uma solução viável em 200s mesmo com bounds razoáveis (~1035 vs. o real).
- **`reference_gurobi.py` em instâncias com janelas de tempo largas**: a fase 1
  (minimizar frota) tem seu próprio orçamento de tempo limitado (no máximo
  ~150s, nunca o `time_limit` inteiro) para não deixar a fase 2 sem tempo — se
  a fase 1 atingir esse limite sem provar otimalidade, o resultado vem com
  `status="fleet_size_not_proven_optimal"` em vez de reportar (incorretamente)
  `"optimal"`. Mesmo com um corte adicional de bin-packing (`fleet_size_lb =
  ceil(demanda_total/capacidade)`), a relaxação MTZ é fraca demais para fechar
  esse gap em instâncias maiores com janelas largas: `c102`/100, por exemplo,
  fica em 11 veículos (bound 10, gap ~9%) mesmo após 300s dedicados só à fase 1
  — o ótimo publicado é 10 veículos/828.94 (igual a `c101`, já que as instâncias
  C1 compartilham coordenadas/demandas e só relaxam as janelas de tempo). Isso é
  um limite de formulação (arc-flow/MTZ compacto), não um bug de orçamento de
  tempo — resolver exatamente exigiria geração de colunas, fora do escopo deste
  baseline.
- **ESPPRC em Python puro**: o *label-setting* elementar (com dominância por
  bitmask de clientes visitados) é implementado do zero e pode ficar lento
  para instâncias com muitos clientes compatíveis entre si (ex.: as
  instâncias `c1xx`, bem "agrupadas"). Não há corte por número de labels/nós
  explorados — só o `time_limit` (relógio) de `lagrangian_relaxation` encerra
  o subproblema; se o *deadline* for atingido antes de provar otimalidade, o
  resultado tem `subproblem_exact=False` e o `lower_bound` reportado **deixa
  de ser um bound certificado** (é só uma aproximação). Para instâncias de 25
  clientes isso costuma acontecer com `time_limit` baixo; aumente o tempo ou
  reduza o número de clientes para bounds exatos.

## Estrutura do projeto

```
src/vrptw/
  instance.py         # parser do formato Solomon + VRPTWInstance
  solution.py          # Route/Solution, validação de viabilidade, custo lexicográfico
  formulation.py       # modelo MILP compacto (gurobipy)
  cuts.py              # separação de RCC e infeasible-path cuts
  solvers.py           # B&B e B&C manuais (árvore best-first própria + Gurobi só como LP)
  reference_gurobi.py  # baseline: MILP inteiro entregue ao solver automático do Gurobi
  lagrangian.py        # ESPPRC (label-setting + ng-route) + subgradiente
scripts/
  solve.py            # CLI: resolve uma instância com um dos métodos (bb/bc/lr/ref)
  run_benchmark.py     # roda os métodos em várias instâncias, salva CSV por instância
  plot_results.py       # gera gráficos comparativos a partir do CSV
tests/                  # pytest (parser, solução, solvers, Lagrangeana)
data/solomon/raw/        # instâncias Solomon de 100 clientes (.txt)
presentation/           # apresentação HTML autocontida (apresentacao.html)
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
