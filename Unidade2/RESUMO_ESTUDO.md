# Resumo de Estudo — Unidade II PCD (OpenMP)

---

## Status de validação das respostas

| Questão | Status | Observação |
|:-------:|:------:|:-----------|
| Q01 | ✅ OK | Dados reais confirmam a análise |
| Q02 | ✅ OK | Conceitos e speedup corretos |
| Q03 | ⚠️ ATENÇÃO | Typo na Tabela 4 (Operando 2 escrito como 10^0, deveria ser 10^3) — cálculo correto |
| Q04 | ✅ OK | Confirmado 100% pelos dados do Santos Dumont |
| Q05 | ✅ OK | Fórmula de Gauss verificada |
| Q07 | ✅ OK | Dados confirmam execução paralela real |
| Q09 | ⚠️ ATENÇÃO | Nota sobre y[1999]/y[2000] está imprecisa (veja detalhes abaixo) |
| Q10 | ✅ OK | Implementação correta e thread-safe |
| Q11 | ✅ OK | Estratégia de histograma local correta |
| Q12 | ✅ OK | Monte Carlo com rand_r e sementes privadas correto |

---

## QUESTÃO 01 — Race Condition (omp_trap_1.c sem critical)

**O que foi respondido:**
- Sem `#pragma omp critical`, threads escrevem simultaneamente em `*global_result_p` → resultado corrompido
- Erros aparecem a partir de **4 threads com n = 10.000** (confirmado pelos dados reais)
- Mais n → mais colisões → mais chance de erro
- Mais threads → mais disputa pela variável → mais chance de erro
- Race condition não é determinística: alguns casos aparecem "corretos por acaso"

**O que estudar / pegadinhas:**
- `critical` é necessário quando múltiplas threads escrevem em variável **compartilhada** (declarada fora do bloco parallel)
- Com **1 thread** nunca há race condition — mas isso não significa que o código está correto para N threads
- O erro **não aparece sempre**: em algumas execuções 2 threads funcionam corretamente → não confunda "funcionou uma vez" com "está correto"
- A variável problema é declarada **fora** do parallel → compartilhada; se fosse dentro → privada → sem race condition
- Alternativas ao `critical`: `reduction`, `atomic`, histograma local

---

## QUESTÃO 02 — Cronometragem com omp_get_wtime()

**O que foi respondido:**
- Com 2 threads: speedup ~2.0× (quase ideal) — padrão "embarrassingly parallel"
- Com 4 threads: speedup ~2.52× (abaixo do ideal por overhead)
- `omp_trap2b.c` tem desempenho equivalente mas código mais limpo (usa `Local_trap()`)
- Speedup não atinge o ideal por: overhead de criação de threads, barreira `#pragma omp barrier`, limitação de banda de memória

**O que estudar / pegadinhas:**
- `omp_get_wtime()` mede **tempo de parede** (wall clock), não tempo de CPU
- Para medir só o bloco paralelo: chame `omp_get_wtime()` **fora** do parallel (antes e depois)
- `#pragma omp barrier` força todas as threads a sincronizarem — tem custo real
- Speedup super-linear (> N×) é raro e indica problemas de medição ou efeito de cache
- `reduction(+:var)` é mais seguro e eficiente que `critical` para somas

---

## QUESTÃO 03 — Aritmética de Ponto Flutuante no Bleeblon (3 dígitos)

**O que foi respondido:**
- Serial: `4 + 3 + 3 + 1000` → soma 10 primeiro, depois 10 + 1000 = 1010 ✓
- Paralelo (2 threads): Thread 0 = 7.0, Thread 1 = 3 + 1000 → **3 é PERDIDO** (1.003×10³ arredonda para 1.00×10³)
- Redução: 7 + 1000 = 1007, arredonda para **1010** — mesmo resultado, por compensação de erros

**⚠️ ERRO encontrado no documento:**
Na Tabela 4 (iteração 4 serial), no passo "Shift one operand", o Operando 2 foi escrito como `1.000 × 10^0` mas deveria ser `1.000 × 10^3`. O **cálculo está correto**, apenas o valor exibido na tabela está errado. Corrija se o professor verificar tabela por tabela.

**O que estudar / pegadinhas:**
- Aritmética de ponto flutuante **NÃO é associativa**: (a+b)+c ≠ a+(b+c) quando há arredondamento
- O paralelismo **muda a ordem das operações** → pode mudar o resultado mesmo que matematicamente equivalente
- Com `reduction`, cada thread soma numa variável local e depois soma os parciais → ordem diferente do serial
- Números com ordens de grandeza muito diferentes causam perda de precisão nos menores
- **Regra de ouro**: somar do menor para o maior reduz erros de arredondamento

---

## QUESTÃO 04 — Escalonamento Padrão de Laços for Paralelos

**O que foi respondido:**
- Escalonamento padrão: **STATIC com chunk = ceil(n / p)**
- Confirmado pelos dados: blocos contíguos, nunca intercalados
- Thread `t` recebe iterações `[t × chunk, min((t+1) × chunk, n) − 1]`
- 100 iterações / 16 threads: 4 primeiras recebem 7, outras 12 recebem 6

**O que estudar / pegadinhas:**
- **STATIC**: divide antes da execução, sem overhead em runtime → ideal para laços uniformes
- **DYNAMIC**: distribui sob demanda em runtime → ideal quando iterações têm custo variável
- **GUIDED**: chunks decrescem ao longo do tempo → equilíbrio entre STATIC e DYNAMIC
- Quando n não é divisível por p: algumas threads recebem 1 iteração a mais (ceil vs floor)
- `schedule(static, k)` define chunk fixo diferente do padrão
- O padrão STATIC é a escolha padrão do OpenMP justamente por ter custo zero em runtime

---

## QUESTÃO 05 — Dependência de Laço (recorrência)

**O que foi respondido:**
- `a[i] = a[i-1] + i` tem dependência de dados → não paralelizável diretamente
- Solução: derivar a **fórmula fechada** `a[i] = i*(i+1)/2` (soma de Gauss)
- Com a fórmula, cada `a[i]` é calculado de forma independente → `#pragma omp parallel for`

**O que estudar / pegadinhas:**
- **Dependência de dados** (data dependency): o valor atual depende do anterior → não paralelizável diretamente
- A chave é identificar se existe uma **fórmula fechada** que elimina a dependência
- Nem todo laço com dependência tem solução paralela trivial — mas recorrências lineares geralmente têm
- `a[i] = a[i-1] + i` expande para `0 + 1 + 2 + ... + i = i*(i+1)/2`
- **Cuidado**: divisão inteira em C: `i*(i+1)/2` funciona pois um dos dois `i` ou `i+1` é sempre par

---

## QUESTÃO 07 — atomic em Variáveis Diferentes

**O que foi respondido:**
- `minha_soma` declarada **dentro** do bloco `parallel` → cada thread tem sua própria cópia privada
- `#pragma omp atomic` em variáveis privadas diferentes → **não forma seção crítica global**
- Confirmado: 8 threads com 8× mais trabalho levam apenas 28% mais tempo (não 800%)
- O small overhead de ~28% é de contenção de cache L3 e overhead de criação de threads, não serialização

**O que estudar / pegadinhas:**
- **critical**: serializa TODAS as threads para qualquer variável → gargalo global
- **atomic**: protege **apenas aquela variável específica** → threads em variáveis diferentes executam em paralelo
- Variável declarada **fora** do parallel → compartilhada → atomic protege de corrida
- Variável declarada **dentro** do parallel → privada → atomic é desnecessário (sem corrida), mas não causa erro
- Se atomic fosse global como critical: 8 threads com 8× trabalho levariam 8× mais tempo → não aconteceu

---

## QUESTÃO 09 — Falso Compartilhamento (False Sharing)

**O que foi respondido:**
- Matriz 8000×8000, 4 threads, cache line = 8 doubles
- Cada thread processa 2000 linhas → acessa y[0..1999], y[2000..3999], y[4000..5999], y[6000..7999]
- Thread 0 vs Thread 2: regiões separadas por 2001 posições → **sem falso compartilhamento** ✓
- Thread 0 vs Thread 3: ainda mais separadas → **sem falso compartilhamento** ✓

**⚠️ IMPRECISÃO encontrada no documento:**
A nota diz que y[1999] e y[2000] "podem estar na mesma cache line". Isso está incorreto: como 2000 é múltiplo de 8, y[1992..1999] forma uma cache line e y[2000..2007] forma a próxima — são cache lines **diferentes**. Falso compartilhamento entre threads adjacentes só ocorreria se o limite caísse no meio de uma cache line (ex: limite em y[1995]).

**O que estudar / pegadinhas:**
- **Falso compartilhamento**: duas threads acessam variáveis **diferentes** que caem na **mesma cache line** → o hardware invalida a linha inteira mesmo sem conflito lógico
- Cache line tem 64 bytes tipicamente = 8 doubles (8 bytes cada)
- Falso compartilhamento só ocorre nas **fronteiras** entre regiões de threads adjacentes, e somente se o limite não for múltiplo de 8
- Solução: padding (alinhar arrays para múltiplos do tamanho da cache line)
- O falso compartilhamento não causa resultados errados — causa apenas **degradação de desempenho**

---

## QUESTÃO 10 — Tokenizador Thread-Safe

**O que foi respondido:**
- `strtok()` usa ponteiro estático interno → não thread-safe (estado global corrompido por 2 threads simultâneas)
- `strtok_r()` é thread-safe mas **modifica a string de entrada** (insere `\0`)
- Solução: `proximo_token()` com ponteiro `pos` local na pilha de cada thread → sem estado global, sem modificar a string

**O que estudar / pegadinhas:**
- **Thread-safe** significa: a função pode ser chamada simultaneamente por múltiplas threads sem produzir resultados incorretos
- Funções com **estado estático interno** nunca são thread-safe
- A solução é passar o estado como parâmetro (ponteiro `pos`) → cada thread tem o seu
- `const char*` garante que a string não será modificada
- `buffer` local na pilha também garante que threads não compartilham espaço de escrita

---

## QUESTÃO 11 — Histograma Paralelo

**O que foi respondido:**
- Estratégia: cada thread aloca seu próprio `local_counts[]` → conta sem contenção
- No final: `#pragma omp critical` para somar ao histograma global (apenas 1× por thread)
- Muito mais eficiente que atomic/critical em cada incremento individual

**O que estudar / pegadinhas:**
- atomic/critical a cada incremento = **serializa todo o laço** → pior que serial
- Histograma local por thread = **zero contenção** durante a fase de contagem
- A redução final com `critical` ocorre apenas **p vezes** (uma por thread) → custo desprezível
- Operação **memory-bound**: percorrer arrays grandes → gargalo de banda de memória, não de CPU
- Com muitas threads em operação memory-bound: saturação do barramento limita speedup

---

## QUESTÃO 12 — Monte Carlo para π com OpenMP

**O que foi respondido:**
- Cada thread gera seus próprios números aleatórios com `rand_r(&semente)` usando semente privada
- `reduction(+:qtd_no_circulo)` acumula os resultados no final
- Algoritmo "embarrassingly parallel": zero comunicação durante os lançamentos
- Precisão: erro reduz por ~√10 a cada 10× mais lançamentos (Lei dos Grandes Números)

**O que estudar / pegadinhas:**
- `rand()` **não é thread-safe** → use `rand_r()` com semente privada por thread
- Sementes diferentes → sequências independentes → sem correlação entre threads
- `reduction(+:var)` é mais limpo e eficiente que `critical` para acumulação numérica
- O método Monte Carlo converge lentamente: precisa de 100M lançamentos para 4 casas decimais
- **Fórmula**: área círculo / área quadrado = π/4 → π = 4 × (pontos dentro do círculo / total)
- Usar `long long` para o contador de pontos (pode passar de 2 bilhões com 100M×8 threads)

---

## Conceitos-chave para não esquecer na prova

| Conceito | Quando usar | Pegadinha clássica |
|----------|-------------|-------------------|
| `#pragma omp critical` | Seção crítica — protege qualquer bloco de código | Serializa TODAS as threads → gargalo |
| `#pragma omp atomic` | Operação atômica simples (+=, -=, etc.) | Só protege aquela variável; variáveis privadas não precisam |
| `reduction(op:var)` | Acumulação paralela (soma, produto, min, max) | Mais eficiente que critical para acumulação |
| `private(var)` | Cada thread tem cópia própria, sem valor inicial | Diferente de `firstprivate` (que copia o valor inicial) |
| `shared(var)` | Variável compartilhada (padrão para vars declaradas fora) | Precisa de sincronização se houver escrita |
| `schedule(static)` | Divide blocos antes de executar | Padrão do OpenMP; ruim se iterações têm custo desigual |
| `schedule(dynamic)` | Distribui iterações sob demanda | Overhead em runtime; bom para laços irregulares |
| `omp_get_wtime()` | Medir tempo de parede | Mede tempo real, não CPU; chame fora do bloco parallel |
| `rand_r(&semente)` | Gerador aleatório thread-safe | `rand()` sozinho não é thread-safe |
| Falso compartilhamento | Cache lines compartilhadas involuntariamente | Não causa erro, causa degradação de desempenho |
