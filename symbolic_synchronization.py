import scipy.io as sp
import matplotlib.pyplot as plt
import numpy as np
import adi
import random

# ---------------------------
# Настройка SDR (Pluto)
# ---------------------------
def standart_settings(Pluto_IP="192.168.3.1", sample_rate=1e6, buffer_size=1e3, gain_mode="manual"):
    """
    Настройка SDR устройства ADALM-Pluto с заданными параметрами.
    Выводится информация о настройках, после чего возвращается объект sdr.
    """
    sdr = adi.Pluto(Pluto_IP)
    sdr.sample_rate = int(sample_rate)
    sdr.rx_buffer_size = int(buffer_size)
    sdr.gain_control_mode_chan0 = gain_mode
    print(f"[INFO] SDR настроен: sample_rate={sample_rate}, buffer_size={buffer_size}, gain_mode={gain_mode}")
    return sdr

# ---------------------------
# Функция отрисовки созвездия (constellation diagram)
# ---------------------------
def plot_constellation(symbols, title="Constellation Diagram"):
    """
    Функция для рисования созвездия (constellation diagram) модулированного сигнала.
    Формируются отдельные фигуры с равными осями, добавляются осевые линии и сетка.
    Здесь plt.show() не вызывается, чтобы все графики можно было вывести одновременно в конце.
    
    Параметры:
        symbols: массив комплексных значений (символы QAM);
        title: заголовок графика.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(np.real(symbols), np.imag(symbols), color='blue', s=50, marker='o', edgecolors='k', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.axhline(0, color='red', linewidth=1, linestyle='--')
    ax.axvline(0, color='red', linewidth=1, linestyle='--')
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    
    max_val = np.max(np.abs(symbols)) * 1.2  # немного больше максимального значения для визуальной наглядности
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    
    plt.tight_layout()
    # Не вызываем plt.show() здесь, чтобы все графики отображались одновременно.

# ---------------------------
# Функция реализующая PLL для коррекции фазовых ошибок
# ---------------------------
def PLL(conv):
    """
    Реализация PLL (Phase-Locked Loop) для коррекции фазовых ошибок.
    PLL – замкнутый контур, сравнивающий фазу входного сигнала с фазой внутреннего генератора,
    корректируя разницу. Здесь используется простая реализация с коэффициентом mu.
    
    Параметры:
        conv: входной сигнал (после временной синхронизации).
    Возвращает:
        Сигнал после фазовой коррекции.
    """
    mu = 0.1  
    theta = 0
    phase_error = np.zeros(len(conv))
    output_signal = np.zeros(len(conv), dtype=np.complex128)
    for n in range(len(conv)):
        theta_hat = np.angle(conv[n])
        # Можно добавить unwrap для устранения скачков с ±π, если необходимо:
        phase_error[n] = theta_hat - theta
        output_signal[n] = conv[n] * np.exp(-1j * theta)
        theta = theta + mu * phase_error[n]
    print(f"[INFO] PLL завершён. Итоговое значение theta = {theta:.3f}")
    return output_signal

# ---------------------------
# Функция реализации Gardner TED для символьной синхронизации
# ---------------------------
def gardner_ted(signal, nsp=10):
    """
    Реализация алгоритма Gardner Timing Error Detector (TED) для символьной синхронизации.
    Алгоритм использует выборки: раннюю, центральную и позднюю, что позволяет оценить
    и скорректировать оптимальный момент выборки символа без точного знания фазы.
    
    Параметры:
        signal: входной сигнал (с повышенным фактором oversampling).
        nsp: количество выборок на символ.
    Возвращает:
        Массив индексов оптимальных точек выборки.
    Также строится график ошибки синхронизации на отдельной фигуре.
    """
    error = np.zeros(len(signal) // nsp)
    index_sync = []
    for i in range(nsp, len(signal) - nsp, nsp):
        s_early = signal[i - nsp//2]
        s_mid = signal[i]
        s_late = signal[i + nsp//2]
        err = (s_late - s_early) * np.conj(s_mid)
        error[i // nsp] = np.real(err)
        index_sync.append(i)
    plt.figure(figsize=(8, 4))
    plt.plot(error, label="Timing Error")
    plt.title("Gardner TED Timing Error")
    plt.xlabel("Index (Symbol periods)")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    print("[INFO] Gardner TED завершён.")
    return np.array(index_sync)

# ---------------------------
# Функция генерации случайного битового потока
# ---------------------------
def randomDataGenerator(size):
    """
    Генерация случайного битового потока заданной длины.
    Выводится количество сгенерированных бит.
    """
    bits = [random.randint(0, 1) for _ in range(size)]
    print(f"[INFO] Сгенерировано {len(bits)} бит.")
    return bits

# ---------------------------
# Функция кодирования 16-QAM (lookup table) по спецификации 3GPP с Gray-кодированием
# ---------------------------
def QAM16(bit_mass):
    """
    Кодирование битовой последовательности в символы 16-QAM согласно спецификации 3GPP с использованием lookup-таблицы.
    Таблица маппинга (нормировка 1/√10 для единичной мощности):
      (0,0,0,0): (-3, -3),  (0,0,0,1): (-3, -1),  (0,0,1,0): (-3,  3), (0,0,1,1): (-3,  1),
      (0,1,0,0): (-1, -3),  (0,1,0,1): (-1, -1),  (0,1,1,0): (-1,  3), (0,1,1,1): (-1,  1),
      (1,0,0,0): ( 3, -3),  (1,0,0,1): ( 3, -1),  (1,0,1,0): ( 3,  3), (1,0,1,1): ( 3,  1),
      (1,1,0,0): ( 1, -3),  (1,1,0,1): ( 1, -1),  (1,1,1,0): ( 1,  3), (1,1,1,1): ( 1,  1).
      
    Масштабирование не применяется здесь, чтобы сохранить корректную структуру созвездия.
    """
    qam16_table = {
        (0, 0, 0, 0): complex(-3, -3) / np.sqrt(10),
        (0, 0, 0, 1): complex(-3, -1) / np.sqrt(10),
        (0, 0, 1, 0): complex(-3,  3) / np.sqrt(10),
        (0, 0, 1, 1): complex(-3,  1) / np.sqrt(10),
        (0, 1, 0, 0): complex(-1, -3) / np.sqrt(10),
        (0, 1, 0, 1): complex(-1, -1) / np.sqrt(10),
        (0, 1, 1, 0): complex(-1,  3) / np.sqrt(10),
        (0, 1, 1, 1): complex(-1,  1) / np.sqrt(10),
        (1, 0, 0, 0): complex( 3, -3) / np.sqrt(10),
        (1, 0, 0, 1): complex( 3, -1) / np.sqrt(10),
        (1, 0, 1, 0): complex( 3,  3) / np.sqrt(10),
        (1, 0, 1, 1): complex( 3,  1) / np.sqrt(10),
        (1, 1, 0, 0): complex( 1, -3) / np.sqrt(10),
        (1, 1, 0, 1): complex( 1, -1) / np.sqrt(10),
        (1, 1, 1, 0): complex( 1,  3) / np.sqrt(10),
        (1, 1, 1, 1): complex( 1,  1) / np.sqrt(10),
    }
    
    if len(bit_mass) % 4 != 0:
        print("QAM16: Ошибка – длина битовой последовательности должна быть кратна 4.")
        raise Exception("QAM16: Длина битовой последовательности должна быть кратна 4.")
    
    sample = np.array([qam16_table[tuple(bit_mass[i:i+4])] for i in range(0, len(bit_mass), 4)])
    print(f"[INFO] QAM16: Сформировано {len(sample)} символов.")
    return sample

def QAM64(bit_mass):
    """
    Кодирование битовой последовательности в символы 64-QAM согласно 3GPP.
    Требуется, чтобы длина bit_mass была кратна 6.
    Маппинг: натуральный порядок, значения вычисляются как 2*index - 7.
    Нормировка: деление на √42, затем масштабирование.
    """
    ampl = 2**14
    if len(bit_mass) % 6 != 0:
        raise Exception("QAM64: Длина битовой последовательности должна быть кратна 6.")
    symbols = np.zeros(len(bit_mass) // 6, dtype=np.complex128)
    for i in range(0, len(bit_mass), 6):
        real_index = bit_mass[i] * 4 + bit_mass[i+1] * 2 + bit_mass[i+2]
        imag_index = bit_mass[i+3] * 4 + bit_mass[i+4] * 2 + bit_mass[i+5]
        real_val = 2 * real_index - 7
        imag_val = 2 * imag_index - 7
        symbols[i // 6] = complex(real_val, imag_val)
    print(f"[INFO] QAM64: Сформировано {len(symbols)} символов.")
    return symbols / np.sqrt(42) * ampl

def QAM256(bit_mass):
    """
    Кодирование битовой последовательности в символы 256-QAM согласно 3GPP.
    Требуется, чтобы длина bit_mass была кратна 8.
    Маппинг: для 4 бит на компоненту, значение вычисляется как 2*index - 15.
    Нормировка: деление на √170, затем масштабирование.
    """
    ampl = 2**14
    if len(bit_mass) % 8 != 0:
        raise Exception("QAM256: Длина битовой последовательности должна быть кратна 8.")
    symbols = np.zeros(len(bit_mass) // 8, dtype=np.complex128)
    for i in range(0, len(bit_mass), 8):
        real_index = bit_mass[i] * 8 + bit_mass[i+1] * 4 + bit_mass[i+2] * 2 + bit_mass[i+3]
        imag_index = bit_mass[i+4] * 8 + bit_mass[i+5] * 4 + bit_mass[i+6] * 2 + bit_mass[i+7]
        real_val = 2 * real_index - 15
        imag_val = 2 * imag_index - 15
        symbols[i // 8] = complex(real_val, imag_val)
    print(f"[INFO] QAM256: Сформировано {len(symbols)} символов.")
    return symbols / np.sqrt(170) * ampl

def QAM1024(bit_mass):
    """
    Кодирование битовой последовательности в символы 1024-QAM согласно 3GPP.
    Требуется, чтобы длина bit_mass была кратна 10.
    Маппинг: для 5 бит на компоненту, значение вычисляется как 2*index - 31.
    Нормировка: деление на √682, затем масштабирование.
    """
    ampl = 2**14
    if len(bit_mass) % 10 != 0:
        raise Exception("QAM1024: Длина битовой последовательности должна быть кратна 10.")
    symbols = np.zeros(len(bit_mass) // 10, dtype=np.complex128)
    for i in range(0, len(bit_mass), 10):
        real_index = (bit_mass[i] * 16 + bit_mass[i+1] * 8 + 
                      bit_mass[i+2] * 4 + bit_mass[i+3] * 2 +
                      bit_mass[i+4])
        imag_index = (bit_mass[i+5] * 16 + bit_mass[i+6] * 8 + 
                      bit_mass[i+7] * 4 + bit_mass[i+8] * 2 +
                      bit_mass[i+9])
        real_val = 2 * real_index - 31
        imag_val = 2 * imag_index - 31
        symbols[i // 10] = complex(real_val, imag_val)
    print(f"[INFO] QAM1024: Сформировано {len(symbols)} символов.")
    return symbols / np.sqrt(682) * ampl

# ---------------------------
# Функции для передачи и приема сигнала через SDR
# ---------------------------
def tx_signal(sdr, tx_lo, gain_tx, data, tx_cycle=True):
    """
    Передача данных через SDR.
    Выводятся параметры передачи, затем выполняется передача.
    """
    sdr.tx_lo = int(tx_lo)
    sdr.tx_hardwaregain_chan0 = gain_tx
    sdr.tx_cyclic_buffer = tx_cycle
    print(f"[INFO] Передача сигнала: tx_lo={tx_lo}, gain_tx={gain_tx}")
    sdr.tx(data)

def rx_signal(sdr, rx_lo, gain_rx, cycle):
    """
    Прием данных через SDR.
    Считывается указанное число циклов, затем объединяются полученные данные.
    """
    sdr.rx_lo = int(rx_lo)
    sdr.rx_hardwaregain_chan0 = gain_rx
    data = []
    for _ in range(cycle):
        rx = sdr.rx()
        data.append(rx)
    print(f"[INFO] Принято {cycle} блоков данных от SDR.")
    return np.concatenate(data)



# ---------------------------
# Основной блок выполнения
# ---------------------------
def main():
    print("=== SDR Алгоритм символьной синхронизации QAM ===")
    sdr = standart_settings("ip:192.168.3.1", 1e6, 10e3)
    
    # Меню выбора схемы модуляции
    print("\nВыберите схему модуляции:")
    print("1: QAM16")
    print("2: QAM64")
    print("3: QAM256")
    print("4: QAM1024")
    choice = input("Введите номер схемы модуляции (по умолчанию 1): ") or "1"
    if choice == "1":
        mod_function = QAM16
        bits_per_symbol = 4
    elif choice == "2":
        mod_function = QAM64
        bits_per_symbol = 6
    elif choice == "3":
        mod_function = QAM256
        bits_per_symbol = 8
    elif choice == "4":
        mod_function = QAM1024
        bits_per_symbol = 10
    else:
        print("[WARN] Неверный выбор, используется QAM16.")
        mod_function = QAM16
        bits_per_symbol = 4

    # Генерация случайной битовой последовательности
    bit_length = 2400
    bit_msg = randomDataGenerator(bit_length)
    valid_length = (len(bit_msg) // bits_per_symbol) * bits_per_symbol
    bit_msg = bit_msg[:valid_length]
    print(f"[INFO] Битовая последовательность длиной {len(bit_msg)} (округлена до кратности {bits_per_symbol}).")
    print("[DEBUG] Первые 20 бит:", bit_msg[:20])
    
    # Кодирование битовой последовательности в QAM-символы
    qam_signal = mod_function(bit_msg)
    print(f"[INFO] Сформировано {len(qam_signal)} QAM символов.")
    
    # Формирование передаваемого сигнала с oversampling (повторами)
    signalRepeat = np.repeat(qam_signal, 10) * (2**14)
    print(f"[INFO] Длина передаваемого сигнала: {len(signalRepeat)} выборок.")
    
    # Передача сигнала через SDR
    tx_signal(sdr, 2300e6, 0, signalRepeat)
    
    # Прием сигнала через SDR и его нормализация
    rx_sig = rx_signal(sdr, 2300e6, 20, 5)
    rx_sig = rx_sig / np.max(np.abs(rx_sig))
    print(f"[INFO] Полученный сигнал нормализован. Длина сигнала: {len(rx_sig)} выборок.")
    
    # Отрисовка графиков созвездия и временной ошибки
    plot_constellation(rx_sig, title="Received Signal")
    gardner_indices = gardner_ted(rx_sig, nsp=10)
    rx_after_ted = rx_sig[gardner_indices]
    print(f"[INFO] После временной синхронизации получено {len(rx_after_ted)} символов.")
    plot_constellation(rx_after_ted, title="After Gardner TED")
    rx_after_pll = PLL(rx_after_ted)
    plot_constellation(rx_after_pll, title="After PLL")
    
    # Вывод всех графиков одновременно
    plt.show()
    
    # Сохранение результатов в MATLAB-файл (при необходимости)
    # sp.savemat('sdr_output.mat', {'rx_signal': rx_sig,
    #                               'after_gardner': rx_after_ted,
    #                               'after_pll': rx_after_pll})
    print("[INFO] Программа завершена.")

if __name__ == '__main__':
    main()