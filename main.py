import torch
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * torch.sqrt(2 * torch.tensor(math.pi)))

def get_mean(array):
    return torch.mean(array)

def get_std_dev(array, mean):
    return torch.sqrt(torch.mean((array - mean) ** 2))

elements_num = 10000
con_prop_first = 0.5
con_prop_second = 0.5

print("Априорная вероятность для первого класса: " + str(con_prop_first))
print("Априорная вероятность для второго класса: " + str(con_prop_second))

array_control = torch.rand(elements_num) * 1100 - 100
array_control.sort()

array_first = torch.rand(elements_num) * 600 - 100
array_second = torch.rand(elements_num) * 600 + 400

mean_first = get_mean(array_first)
mean_second = get_mean(array_second)

std_dev_first = get_std_dev(array_first, mean_first)
std_dev_second = get_std_dev(array_second, mean_second)

p_first = gaussian(torch.arange(1000).float(), mean_first, std_dev_first) * con_prop_first
p_second = gaussian(torch.arange(1000).float(), mean_second, std_dev_second) * con_prop_second

prop_false_alarm = 0
prop_skip_detect = 0
i = 0

while p_first[i] > p_second[i]:
    prop_false_alarm += p_second[i]
    i += 1

plt.axvline(x=i, c='m')

while i < 1000:
    prop_skip_detect += p_first[i]
    i += 1

print("Вероятность ложной тревоги: " + str(prop_skip_detect.item()))
print("Вероятность пропуска обнаружения ошибки: " + str(prop_false_alarm.item()))
print("Суммарная ошибка классификации: " + str((prop_false_alarm + prop_skip_detect).item()))

plt.plot(p_first.numpy(), c='b')
plt.plot(p_second.numpy(), c='c')
plt.show()
