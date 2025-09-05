iimport streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Necesitas instalar deap si no lo tienes: pip install deap
from deap import base, creator, tools, algorithms

# ----------------------- Configuración Inicial DEAP -----------------------
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

# ----------------------- Funciones de Carga y Procesamiento de Datos -----------------------

@st.cache_data
def load_data_and_counts(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # AHORA REQUIERE 'Frecuencia'
            if 'Numero' in df.columns and 'Atraso' in df.columns and 'Frecuencia' in df.columns:
                df['Numero'] = pd.to_numeric(df['Numero'], errors='coerce')
                df['Atraso'] = pd.to_numeric(df['Atraso'], errors='coerce')
                df['Frecuencia'] = pd.to_numeric(df['Frecuencia'], errors='coerce')
                df.dropna(subset=['Numero', 'Atraso', 'Frecuencia'], inplace=True)
                df['Numero'] = df['Numero'].astype(int).astype(str)
                df['Atraso'] = df['Atraso'].astype(int)
                df['Frecuencia'] = df['Frecuencia'].astype(int)

                if df.empty:
                     st.warning("El archivo no contiene filas válidas.")
                     return None, {}, {}, {}, [], {}, 0, {}

                st.success("Archivo de datos cargado exitosamente.")
                numero_a_atraso = dict(zip(df['Numero'], df['Atraso']))
                numero_a_frecuencia = dict(zip(df['Numero'], df['Frecuencia']))
                
                atrasos_disponibles_int = sorted(list(set(df['Atraso'].tolist())))
                numeros_validos = list(numero_a_atraso.keys())
                prob_por_numero = 1.0 / len(numeros_validos) if numeros_validos else 0
                distribucion_probabilidad = {num: prob_por_numero for num in numeros_validos}
                atraso_counts = df['Atraso'].astype(str).value_counts().to_dict()
                total_atraso_dataset = df['Atraso'].sum()

                atraso_stats = {
                    "min": df['Atraso'].min(), "max": df['Atraso'].max(), "mean": df['Atraso'].mean(),
                    "p25": df['Atraso'].quantile(0.25), "p75": df['Atraso'].quantile(0.75)
                }
                return df, numero_a_atraso, numero_a_frecuencia, distribucion_probabilidad, atrasos_disponibles_int, atraso_counts, total_atraso_dataset, atraso_stats
            else:
                 st.error("El archivo debe contener las columnas 'Numero', 'Atraso' y 'Frecuencia'.")
                 return None, {}, {}, {}, [], {}, 0, {}
        except Exception as e:
            st.error(f"Error al procesar el archivo de datos: {e}")
            return None, {}, {}, {}, [], {}, 0, {}
    return None, {}, {}, {}, [], {}, 0, {}

@st.cache_data
def load_historical_combinations(uploaded_file):
    if uploaded_file is not None:
        try:
            df_hist = pd.read_csv(uploaded_file, header=None)
            historical_sets = [set(pd.to_numeric(row, errors='coerce').dropna().astype(int)) for _, row in df_hist.iterrows()]
            historical_sets = [s for s in historical_sets if len(s) >= 6]
            if not historical_sets:
                st.warning("El archivo de historial no contenía combinaciones válidas.")
                return []
            st.success(f"Archivo de historial cargado: {len(historical_sets)} combinaciones encontradas.")
            return historical_sets
        except Exception as e:
            st.error(f"Error al procesar el archivo de historial: {e}")
            return []
    return []

# --- Funciones de Análisis Histórico ---
@st.cache_data
def analyze_historical_special_calc(historical_sets, total_atraso_dataset, numero_a_atraso):
    if not historical_sets or total_atraso_dataset is None: return None
    values = [total_atraso_dataset + 40 - sum(numero_a_atraso.get(str(num), 0) for num in s) for s in historical_sets]
    if not values: return None
    return {"min": int(np.min(values)), "max": int(np.max(values)), "mean": int(np.mean(values)), "std": int(np.std(values))}

@st.cache_data
def analyze_historical_frequency_sum(historical_sets, numero_a_frecuencia):
    if not historical_sets or not numero_a_frecuencia: return None
    values = [sum(numero_a_frecuencia.get(str(num), 0) for num in s) for s in historical_sets]
    if not values: return None
    return {"min": int(np.min(values)), "max": int(np.max(values)), "mean": int(np.mean(values)), "std": int(np.std(values))}

@st.cache_data
def analyze_historical_composition(historical_sets, numero_a_atraso, composicion_ranges):
    if not historical_sets: return None
    def get_category(atraso, ranges):
        if ranges['caliente'][0] <= atraso <= ranges['caliente'][1]: return 'caliente'
        elif ranges['tibio'][0] <= atraso <= ranges['tibio'][1]: return 'tibio'
        elif ranges['frio'][0] <= atraso <= ranges['frio'][1]: return 'frio'
        elif atraso >= ranges['congelado'][0]: return 'congelado'
        return 'otro'
    counts = Counter(tuple(Counter(get_category(numero_a_atraso.get(str(num), -1), composicion_ranges) for num in s).get(cat, 0) for cat in ['caliente', 'tibio', 'frio', 'congelado']) for s in historical_sets)
    return counts if counts else None

# --- Motores de Generación y Filtrado ---

def generar_combinaciones_con_restricciones(dist_prob, num_a_atraso, num_a_freq, restr_atraso, n_sel, n_comb, hist_combs, total_atraso, special_range, freq_sum_range):
    valores = list(dist_prob.keys())
    combinaciones = []
    intentos = 0
    max_intentos = n_comb * 300
    while len(combinaciones) < n_comb and intentos < max_intentos:
        intentos += 1
        seleccionados = random.sample(valores, n_sel)
        
        suma_frecuencias = sum(num_a_freq.get(val, 0) for val in seleccionados)
        if not (freq_sum_range[0] <= suma_frecuencias <= freq_sum_range[1]):
            continue

        if any(Counter(num_a_atraso.get(n, -1) for n in seleccionados)[int(a)] > l for a, l in restr_atraso.items()): continue
        suma_atrasos = sum(num_a_atraso.get(val, 0) for val in seleccionados)
        valor_especial = total_atraso + 40 - suma_atrasos
        if not (special_range[0] <= valor_especial <= special_range[1]): continue
        if hist_combs and any(len(set(int(n) for n in seleccionados).intersection(h)) > 2 for h in hist_combs): continue
        
        seleccionados.sort(key=int)
        combinaciones.append(tuple(seleccionados))
        
    conteo = Counter(combinaciones)
    return sorted({c: (f, np.prod([dist_prob.get(v, 0) for v in c])) for c, f in conteo.items()}.items(), key=lambda x: (-x[1][1], -x[1][0]))

def procesar_combinaciones(dist_prob, num_a_atraso, num_a_freq, restr_atraso, n_sel, n_comb, n_ejec, hist_combs, total_atraso, special_range, freq_sum_range):
    resultados = []
    args = (dist_prob, num_a_atraso, num_a_freq, restr_atraso, n_sel, n_comb, hist_combs, total_atraso, special_range, freq_sum_range)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generar_combinaciones_con_restricciones, *args) for _ in range(n_ejec)]
        for future in as_completed(futures):
            try: resultados.append(future.result())
            except Exception as e: st.error(f"Un proceso falló: {e}")
    return resultados

def filtrar_por_composicion(combinaciones, numero_a_atraso, composicion_rules):
    def get_category(atraso, ranges):
        if ranges['caliente'][0] <= atraso <= ranges['caliente'][1]: return 'caliente'
        elif ranges['tibio'][0] <= atraso <= ranges['tibio'][1]: return 'tibio'
        elif ranges['frio'][0] <= atraso <= ranges['frio'][1]: return 'frio'
        elif atraso >= ranges['congelado'][0]: return 'congelado'
        return 'otro'
    combinaciones_filtradas = [comb for comb in combinaciones if all(Counter(get_category(numero_a_atraso.get(str(num), -1), composicion_rules['ranges']) for num in comb).get(cat, 0) == count for cat, count in composicion_rules['counts'].items())]
    return combinaciones_filtradas

def evaluar_individuo_deap(individuo, dist_prob, num_a_atraso, num_a_freq, restr_atraso, n_sel, hist_combs, total_atraso, special_range, freq_sum_range):
    if len(individuo) != n_sel or len(set(individuo)) != n_sel: return (0,)
    
    suma_frecuencias = sum(num_a_freq.get(val, 0) for val in individuo)
    if not (freq_sum_range[0] <= suma_frecuencias <= freq_sum_range[1]): return (0,)
        
    atrasos = Counter(num_a_atraso.get(val) for val in individuo)
    if any(cantidad > restr_atraso.get(str(atraso), n_sel) for atraso, cantidad in atrasos.items()): return (0,)
    if hist_combs and any(len(set(int(n) for n in individuo).intersection(h)) > 2 for h in hist_combs): return (0,)
    suma_atrasos = sum(num_a_atraso.get(val, 0) for val in individuo)
    valor_especial = total_atraso + 40 - suma_atrasos
    if not (special_range[0] <= valor_especial <= special_range[1]): return (0,)
    
    return (np.prod([dist_prob.get(val, 0) for val in individuo]),)

def ejecutar_algoritmo_genetico(n_gen, n_pob, cxpb, mutpb, dist_prob, num_a_atraso, num_a_freq, restr_atraso, hist_combs, total_atraso, special_range, freq_sum_range, n_sel=6):
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, list(dist_prob.keys()), n_sel)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluar_individuo_deap, dist_prob=dist_prob, num_a_atraso=num_a_atraso, num_a_freq=num_a_freq, restr_atraso=restr_atraso, n_sel=n_sel, hist_combs=hist_combs, total_atraso=total_atraso, special_range=special_range, freq_sum_range=freq_sum_range)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=n_pob)
    algorithms.eaSimple(pop, toolbox, cxpb, mutpb, n_gen, verbose=False)
    best = tools.selBest(pop, k=1)[0] if pop else None
    if best:
        best = sorted(list(set(best)), key=int)
        if len(best) != n_sel: return None, 0.0, "AG no mantuvo individuos válidos."
        return best, toolbox.evaluate(best)[0], None
    return None, 0.0, "Población final vacía."

# ----------------------- Interfaz Gráfica de Streamlit -----------------------

st.set_page_config(layout="wide", page_title="Generador de Combinaciones Homeostático")
st.title("Modelo Homeostático de Generación de Combinaciones")

if 'suggested_composition' not in st.session_state:
    st.session_state.suggested_composition = None

st.header("1. Cargar Archivos de Datos")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Sube CSV de Datos ('Numero', 'Atraso', 'Frecuencia')", type="csv", key="data_uploader")
    df, num_a_atraso, num_a_freq, dist_prob, atrasos_disp, atraso_counts, total_atraso, atraso_stats = load_data_and_counts(uploaded_file)
with col2:
    hist_file = st.file_uploader("Sube CSV con Historial de Combinaciones (Opcional)", type="csv", key="history_uploader")
    historical_combinations_set = load_historical_combinations(hist_file)

n_selecciones = 6
if df is not None:
     st.info(f"**Suma total de 'Atraso' en el dataset:** {total_atraso}")

st.header("2. Configurar Parámetros y Restricciones")
restricciones_finales, composicion_rules = {}, {}
total_count_composition = 0
special_calc_range, freq_sum_range = (0, 99999), (0, 99999)

if df is not None:
    with st.expander("Configurar Restricciones de Atraso (Etapa 1)"):
        selected_atrasos = st.multiselect("Selecciona 'Atraso' a restringir:", options=[str(a) for a in atrasos_disp], default=[str(a) for a in atrasos_disp])
        cols = st.columns(4)
        for i, atraso_str in enumerate(selected_atrasos):
            with cols[i % 4]:
                limit = st.number_input(f"Max Atraso '{atraso_str}':", 0, n_selecciones, atraso_counts.get(atraso_str, 0), key=f"res_{atraso_str}")
                restricciones_finales[atraso_str] = limit
    
    if historical_combinations_set:
        with st.expander("Filtro por Suma de Frecuencias (Etapa 1)", expanded=False):
            stats_freq = analyze_historical_frequency_sum(historical_combinations_set, num_a_freq)
            if stats_freq:
                st.info(f"Análisis del historial: La 'Suma de Frecuencias' varía de **{stats_freq['min']}** a **{stats_freq['max']}**, con un promedio de **{stats_freq['mean']}**.")
                default_range_freq = (stats_freq['mean'] - stats_freq['std'], stats_freq['mean'] + stats_freq['std'])
                freq_sum_range = st.slider("Rango deseado para 'Suma de Frecuencias':", min_value=stats_freq['min'] - 20, max_value=stats_freq['max'] + 20, value=default_range_freq, key="freq_slider")

        with st.expander("Filtro de 'Cálculo Especial' (Etapa 1)", expanded=False):
            stats_special = analyze_historical_special_calc(historical_combinations_set, total_atraso, num_a_atraso)
            if stats_special:
                st.info(f"Análisis del historial: 'Cálculo Especial' varía de **{stats_special['min']}** a **{stats_special['max']}**, promedio **{stats_special['mean']}**.")
                default_range_special = (stats_special['mean'] - stats_special['std'], stats_special['mean'] + stats_special['std'])
                special_calc_range = st.slider("Rango deseado:", stats_special['min'] - 50, stats_special['max'] + 50, value=default_range_special, key="special_slider")

    with st.expander("Filtro de Composición de Atrasos (Etapa 2)", expanded=True):
        st.info("Define rangos, analiza el historial y especifica la composición deseada para la Etapa 2.")
        
        st.subheader("2.1. Definir Categorías de Atraso")
        max_atraso = atraso_stats.get("max", 100)
        c1, c2 = st.columns(2)
        with c1:
            range_caliente = st.slider("Rango 'Caliente'", 0, max_atraso, (0, int(atraso_stats.get("p25", 5))), key="r_hot")
            range_frio = st.slider("Rango 'Frío'", 0, max_atraso, (int(atraso_stats.get("p75", 15)), max_atraso - 1), key="r_cold")
        with c2:
            range_tibio = st.slider("Rango 'Tibio'", 0, max_atraso, (range_caliente[1] + 1, range_frio[0] -1), key="r_warm")
            min_congelado = st.number_input("Atraso mínimo 'Congelado'", value=max_atraso, key="r_icy")
        
        current_ranges = {'caliente': range_caliente, 'tibio': range_tibio, 'frio': range_frio, 'congelado': (min_congelado, 9999)}
        
        if historical_combinations_set:
            st.subheader("2.2. Análisis de Composición Histórica")
            comp_analysis = analyze_historical_composition(historical_combinations_set, num_a_atraso, current_ranges)
            if comp_analysis:
                most_common = comp_analysis.most_common(5)
                most_common_comp, freq = most_common[0]
                
                st.success(f"**Composición más frecuente:** {most_common_comp[0]} Calientes, {most_common_comp[1]} Tibios, {most_common_comp[2]} Fríos, {most_common_comp[3]} Congelados")
                if st.button("Aplicar Composición Sugerida"):
                    st.session_state.suggested_composition = most_common_comp
                    st.rerun()
                st.dataframe(pd.DataFrame(most_common, columns=['Composición (C-T-F-C)', 'Frecuencia']).assign(Probabilidad=lambda x: (x['Frecuencia'] / sum(comp_analysis.values())) * 100))
        
        st.subheader("2.3. Especificar Composición Deseada")
        suggested = st.session_state.suggested_composition
        c3, c4, c5, c6 = st.columns(4)
        with c3: count_caliente = st.number_input("Nº Calientes", 0, n_selecciones, suggested[0] if suggested else 2, key="c_hot")
        with c4: count_tibio = st.number_input("Nº Tibios", 0, n_selecciones, suggested[1] if suggested else 2, key="c_warm")
        with c5: count_frio = st.number_input("Nº Fríos", 0, n_selecciones, suggested[2] if suggested else 2, key="c_cold")
        with c6: count_congelado = st.number_input("Nº Congelados", 0, n_selecciones, suggested[3] if suggested else 0, key="c_icy")

        total_count_composition = count_caliente + count_tibio + count_frio + count_congelado
        if total_count_composition == n_selecciones:
            st.success("La composición es válida.")
            composicion_rules = {'ranges': current_ranges, 'counts': {'caliente': count_caliente, 'tibio': count_tibio, 'frio': count_frio, 'congelado': count_congelado}}
        else: st.warning(f"La suma de las cantidades ({total_count_composition}) debe ser igual a {n_selecciones}.")
    
    with st.expander("Configurar Parámetros de los Algoritmos"):
        col_ga, col_sim = st.columns(2)
        with col_ga: st.subheader("Algoritmo Genético"); ga_ngen=st.slider("Generaciones",10,1000,200); ga_npob=st.slider("Población",100,5000,1000); ga_cxpb=st.slider("Cruce",0.0,1.0,0.7); ga_mutpb=st.slider("Mutación",0.0,1.0,0.2)
        with col_sim: st.subheader("Simulación en Cascada"); sim_n_comb=st.number_input("Combinaciones/Ejec.",1000,value=50000); sim_n_ejec=st.number_input("Ejecuciones",1,value=8)
else:
    st.info("Carga un archivo de Datos ('Numero', 'Atraso', 'Frecuencia') para empezar.")

st.header("3. Ejecutar Algoritmos")
if df is None:
    st.warning("Carga un archivo de Datos para poder ejecutar los algoritmos.")
else:
    run_col1, run_col2 = st.columns(2)
    with run_col1:
        if st.button("Ejecutar Algoritmo Genético"):
            with st.spinner("Buscando la mejor combinación..."):
                mejor_ind, mejor_fit, err_msg = ejecutar_algoritmo_genetico(ga_ngen, ga_npob, ga_cxpb, ga_mutpb, dist_prob, num_a_atraso, num_a_freq, restricciones_finales, historical_combinations_set, total_atraso, special_calc_range, freq_sum_range, n_selecciones)
            if err_msg: st.error(err_msg)
            elif mejor_ind and mejor_fit > 0:
                st.subheader("Mejor Combinación (GA)"); st.success(f"**Combinación: {' - '.join(map(str, mejor_ind))}**")
                st.write(f"**Suma Frecuencias:** {sum(num_a_freq.get(v, 0) for v in mejor_ind)}")
                st.write(f"**Cálculo Especial:** {total_atraso + 40 - sum(num_a_atraso.get(v, 0) for v in mejor_ind)}")
            else: st.warning("El GA no encontró una combinación válida con las restricciones.")

    with run_col2:
        if st.button("Ejecutar Simulación en Cascada"):
            with st.spinner("Etapa 1: Generando combinaciones en paralelo..."):
                start_time = time.time()
                resultados = procesar_combinaciones(dist_prob, num_a_atraso, num_a_freq, restricciones_finales, n_selecciones, sim_n_comb, sim_n_ejec, historical_combinations_set, total_atraso, special_calc_range, freq_sum_range)
                st.info(f"Etapa 1 completada en {time.time() - start_time:.2f} segundos.")
            
            todas_unicas = list(set(tuple(int(n) for n in comb) for res in resultados for comb, _ in res))
            st.info(f"Se generaron **{len(todas_unicas)}** combinaciones únicas en la Etapa 1.")

            combinaciones_refinadas = []
            if total_count_composition == n_selecciones:
                with st.spinner("Etapa 2: Aplicando filtro de composición..."):
                    combinaciones_refinadas = filtrar_por_composicion(todas_unicas, num_a_atraso, composicion_rules)
                st.success(f"Etapa 2 completada. Se encontraron **{len(combinaciones_refinadas)}** combinaciones que cumplen el perfil.")
            else:
                st.error("No se ejecutó Etapa 2: la suma de cantidades de composición no es correcta.")

            st.subheader(f"Resultados del Filtro de Composición (Etapa 2) - {len(combinaciones_refinadas)} combinaciones")
            if combinaciones_refinadas:
                data = [{"Combinación": " - ".join(map(str, sorted(c))), "Suma Frecuencias": sum(num_a_freq.get(str(v),0) for v in c), "Suma Atrasos": sum(num_a_atraso.get(str(v),0) for v in c), "Cálculo Especial": total_atraso + 40 - sum(num_a_atraso.get(str(v),0) for v in c)} for c in combinaciones_refinadas]
                st.dataframe(pd.DataFrame(data).reset_index(drop=True))
            else:
                st.info("Ninguna combinación superó el filtro de la Etapa 2.")

st.sidebar.header("Guía del Modelo")
st.sidebar.markdown("""
Esta aplicación genera combinaciones basadas en el **principio de homeostasis**, que asume que un sistema aleatorio tiende a mantenerse en equilibrio.

**Filtros de Homeostasis (Etapa 1):**
1.  **Suma de Frecuencias:** Controla el equilibrio a largo plazo.
2.  **Cálculo Especial:** Controla el equilibrio a corto plazo (basado en atrasos).
3.  **Restricciones de Atraso:** Evita concentraciones anómalas de atrasos individuales.

**Filtro Estratégico (Etapa 2):**
- **Composición:** Define la "personalidad" de la combinación (ej. 2 'Calientes', 3 'Fríos', etc.). La app **recomienda la estrategia más común** del historial.
""")
