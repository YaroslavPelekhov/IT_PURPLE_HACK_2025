import json
import argparse


hosts = {}               # { host_name: {"cpu": int, "ram": int} }
allocation = {}          # { host_name: [vm_name, ...] }
vm_to_host = {}          # { vm_name: host_name }
used_resources = {}      # { host_name: {"cpu": int, "ram": int} }
vm_specs = {}            # { vm_name: {"cpu": int, "ram": int} }
round_counter = 0        # Переменная для отслеживания текущего раунда
host_count = 0           # Количество хостов


def util_calculate(host, host_cap, vm_cpu, vm_ram):
    cpu = vm_cpu + used_resources[host]["cpu"]
    ram = vm_ram + used_resources[host]["ram"]
    cpu_util = cpu / host_cap["cpu"]
    ram_util = ram / host_cap["ram"]
    return max(cpu_util, ram_util)

def try_place_vm(vm, specs, migrations, evenly_distribute=False):
    vm_cpu = specs["cpu"]
    vm_ram = specs["ram"]

    if evenly_distribute:
        # Для равномерного распределения выбираем хост с минимальным числом размещённых ВМ
        least_loaded_host = None
        least_vm_count = float("inf")
        for host in hosts.keys():
            if len(allocation[host]) < least_vm_count:
                least_loaded_host = host
                least_vm_count = len(allocation[host])
        if least_loaded_host:
            allocation[least_loaded_host].append(vm)
            vm_to_host[vm] = least_loaded_host
            used_resources[least_loaded_host]["cpu"] += vm_cpu
            used_resources[least_loaded_host]["ram"] += vm_ram
            return True, least_loaded_host

    # В остальных случаях выполняется обычное размещение по утилизации
    best_util = 1
    best_host = None
    for host, cap in hosts.items():
        avail_cpu = cap["cpu"] - used_resources[host]["cpu"]
        avail_ram = cap["ram"] - used_resources[host]["ram"]
        if avail_cpu >= vm_cpu and avail_ram >= vm_ram:
            util = util_calculate(host, cap, vm_cpu, vm_ram)
            if util > 0.85:
                ut = abs(util - 0.807) + 0.1
            else:
                ut = abs(util - 0.807)
            if ut < best_util:
                best_host = host
                best_util = ut

    if best_host:
        allocation[best_host].append(vm)
        vm_to_host[vm] = best_host
        used_resources[best_host]["cpu"] += vm_cpu
        used_resources[best_host]["ram"] += vm_ram
        return True, best_host

    # Если прямого размещения нет, пытаемся провести миграцию
    for host, cap in hosts.items():
        avail_cpu = cap["cpu"] - used_resources[host]["cpu"]
        avail_ram = cap["ram"] - used_resources[host]["ram"]
        for migrating_vm in list(allocation[host]):  # копия списка для безопасного удаления
            mig_specs = vm_specs[migrating_vm]
            freed_cpu = mig_specs["cpu"]
            freed_ram = mig_specs["ram"]
            if avail_cpu + freed_cpu >= vm_cpu and avail_ram + freed_ram >= vm_ram:
                for target_host, tcap in hosts.items():
                    if target_host == host:
                        continue
                    targ_avail_cpu = tcap["cpu"] - used_resources[target_host]["cpu"]
                    targ_avail_ram = tcap["ram"] - used_resources[target_host]["ram"]
                    if targ_avail_cpu >= mig_specs["cpu"] and targ_avail_ram >= mig_specs["ram"]:
                        allocation[host].remove(migrating_vm)
                        allocation[target_host].append(migrating_vm)
                        vm_to_host[migrating_vm] = target_host
                        used_resources[host]["cpu"] -= mig_specs["cpu"]
                        used_resources[host]["ram"] -= mig_specs["ram"]
                        used_resources[target_host]["cpu"] += mig_specs["cpu"]
                        used_resources[target_host]["ram"] += mig_specs["ram"]
                        migrations[migrating_vm] = {"from": host, "to": target_host}
                        avail_cpu = cap["cpu"] - used_resources[host]["cpu"]
                        avail_ram = cap["ram"] - used_resources[host]["ram"]
                        if avail_cpu >= vm_cpu and avail_ram >= vm_ram:
                            allocation[host].append(vm)
                            vm_to_host[vm] = host
                            used_resources[host]["cpu"] += vm_cpu
                            used_resources[host]["ram"] += vm_ram
                            return True, host
    return False, None


def optimize_allocations():
    # Функция перемещает не более одной ВМ (можно изменить лимит, если потребуется)
    # только если миграция приближает утилизацию одного из задействованных хостов (источника или приёмника)
    # к значению 0.807 или если перенос полностью освобождает хост-источник.
    migrations = {}
    restricted_pairs = set()  # Хранит пары (source, destination), чтобы запретить обратную миграцию
    host_utilizations = []
    if (round_counter + 1) % 2:
        return  migrations

    # Для сортировки рассматриваем разницу текущей утилизации от целевого значения 0.807
    for host, cap in hosts.items():
        if used_resources[host]["cpu"] == 0 and used_resources[host]["ram"] == 0:
            util_diff = 0  # пустой хост
        else:
            util = max(
                used_resources[host]["cpu"] / cap["cpu"],
                used_resources[host]["ram"] / cap["ram"]
            )
            util_diff = abs(util - 0.807)
        host_utilizations.append((host, util_diff))
    # Сначала рассматриваем хосты, у которых разница максимальна – больше возможностей для улучшения
    host_utilizations.sort(key=lambda x: -x[1])

    migration_count = 0
    # Перебираем хосты-источники (source hosts) для возможной миграции ВМ
    for source, _ in host_utilizations:
        # Ограничение количества миграций за раунд (можно настроить)
        if migration_count >= round_counter % 2:
            break
        # Для каждого хоста пробуем рассмотреть миграцию каждой ВМ
        for vm in list(allocation[source]):
            specs = vm_specs[vm]
            best_improvement = None
            best_target = None
            # Рассчитываем текущую утилизацию источника
            source_cap = hosts[source]
            source_util = max(used_resources[source]["cpu"] / source_cap["cpu"],
                              used_resources[source]["ram"] / source_cap["ram"])
            # Предполагаемая утилизация источника после удаления ВМ
            new_source_util = max((used_resources[source]["cpu"] - specs["cpu"]) / source_cap["cpu"],
                                  (used_resources[source]["ram"] - specs["ram"]) / source_cap["ram"])
            # Флаг, освобождается ли источник после удаления ВМ
            frees_source = (len(allocation[source]) == 1)

            # Перебираем возможные целевые хосты для миграции
            for target, target_cap in hosts.items():
                if target == source:
                    continue
                # Пропускаем, если характеристики хостов идентичны – миграция между ними не рассматривается
                if hosts[source] == hosts[target]:
                    continue
                # Запрещаем обратную миграцию при наличии пары
                if (target, source) in restricted_pairs or (source, target) in restricted_pairs:
                    continue
                # Проверяем, хватает ли ресурсов на целевом хосте для размещения ВМ
                available_cpu = target_cap["cpu"] - used_resources[target]["cpu"]
                available_ram = target_cap["ram"] - used_resources[target]["ram"]
                if available_cpu < specs["cpu"] or available_ram < specs["ram"]:
                    continue
                # Рассчитываем текущую утилизацию целевого хоста и предполагаемую после добавления ВМ
                target_util = max(used_resources[target]["cpu"] / target_cap["cpu"],
                                  used_resources[target]["ram"] / target_cap["ram"])
                new_target_util = max((used_resources[target]["cpu"] + specs["cpu"]) / target_cap["cpu"],
                                      (used_resources[target]["ram"] + specs["ram"]) / target_cap["ram"])
                if target_util > 0.807:
                    target_util = (target_util - 0.807) * 4
                if new_target_util > 0.807:
                    new_target_util = (new_target_util - 0.807) * 4

                # Определяем, будет ли миграция улучшением для источника или для приёмника
                improve_source = (abs(new_source_util - 0.807) + 0.15 < abs(source_util - 0.807)) or frees_source
                improve_target = (abs(new_target_util - 0.807) + 0.15 < abs(target_util - 0.807)) or frees_source
                # Если ни у источника, ни у приёмника улучшения не наблюдается, миграция нецелесообразна
                if not (improve_source and improve_target):
                    continue
                # Можно оценить суммарное улучшение как сумму уменьшений отклонения для источника и приёмника
                improvement = (abs(source_util - 0.807) - abs(new_source_util - 0.807)) + (
                            abs(target_util - 0.807) - abs(new_target_util - 0.807))
                # Выбираем целевой хост с наибольшим улучшением
                if best_improvement is None or improvement > best_improvement:
                    best_improvement = improvement
                    best_target = target

            # Если найден подходящий целевой хост, выполняем миграцию
            if best_target:
                allocation[source].remove(vm)
                allocation[best_target].append(vm)
                vm_to_host[vm] = best_target
                # Обновляем задействованные ресурсы для обоих хостов
                used_resources[source]["cpu"] -= specs["cpu"]
                used_resources[source]["ram"] -= specs["ram"]
                used_resources[best_target]["cpu"] += specs["cpu"]
                used_resources[best_target]["ram"] += specs["ram"]
                migrations[vm] = {"from": source, "to": best_target}
                restricted_pairs.add((source, best_target))
                migration_count += 1
                break  # После миграции ВМ с данного хоста переходим к следующему источнику
    return migrations


def process_round(input_data):
    global hosts, allocation, vm_to_host, used_resources, vm_specs, round_counter
    migrations = {}
    allocation_failures = []
    round_counter += 1

    if not hosts:
        global host_count
        hosts = input_data.get("hosts", {})
        for host, cap in hosts.items():
            used_resources[host] = {"cpu": 0, "ram": 0}
            allocation[host] = []
            host_count += 1


    current_vms = input_data.get("virtual_machines", {})
    new_vm_set = set(current_vms.keys())
    old_vm_set = set(vm_specs.keys())
    removed_vms = old_vm_set - new_vm_set

    for vm in removed_vms:
        if vm in vm_to_host:
            host = vm_to_host[vm]
            specs = vm_specs[vm]
            used_resources[host]["cpu"] -= specs["cpu"]
            used_resources[host]["ram"] -= specs["ram"]
            allocation[host].remove(vm)
            vm_to_host.pop(vm)
            vm_specs.pop(vm)

    added_vms = new_vm_set - old_vm_set
    
    for vm in added_vms:
        specs = current_vms[vm]
        vm_specs[vm] = specs
        evenly_distribute = round_counter <= (host_count % 10) % host_count + 5
        placed, _ = try_place_vm(vm, specs, migrations, evenly_distribute=evenly_distribute)
        if not placed:
            allocation_failures.append(vm)

    for vm, specs in current_vms.items():
        if vm not in vm_specs:
            vm_specs[vm] = specs
    if round_counter > (host_count % 10) % host_count + 5:
        migrations.update(optimize_allocations())

    output = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "allocations": allocation,
        "allocation_failures": allocation_failures,
        "migrations": migrations
    }

    
    return output

def main(input_line):
    try:
        input_data = json.loads(input_line)
    except json.JSONDecodeError as e:
        print("Ошибка при декодировании JSON:", e)
        return
    result = process_round(input_data=input_data)
    
    print(json.dumps(result, ensure_ascii=False))



while True:
    try:
        line = input()  # Ввод одной строки
        if line.strip() == "":
            break
        main(line)
    except EOFError:
        break
