import time
import File_processing as fp
import Connectivity_repair_SP as sp


#CONSTANTES
Rc = 6
Rs = 4
Ru = 2.5
sensitivity = -94


def save_results(fichier, result):
    with open(fichier, 'a') as f:
        # Write a new line to the file
        f.write(result)
        f.write('\n')



def init_archi1():
    nb_zones = 15 * 5 + 5 * 2
    nb_targets = 29
    file1 = open('targets_archi1', 'r')
    list_target_points = [int(x) for x in file1.readline().split(',')]
    obstacles = fp.load_obstacles("archi1")
    #print("finish loading obstacles")
    coordinates = fp.generate_coordinates_archi1()
    #print("finish generating coordinates")
    #t = time.time()
    communication_graph = sp.generate_list_connections_between_positions(coordinates, obstacles, sensitivity, Rc)
    #print("finish generating graph of zones in ", end=" ")
    #print(time.time() - t, end=" s")
    return nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles



def init_archi2():
    nb_zones = 60 * 30 + 5 * 20
    nb_targets = 665
    file1 = open('targets_archi2', 'r')
    list_target_points = [int(x) for x in file1.readline().split(',')]
    obstacles = fp.load_obstacles("archi2")
    print("finish loading obstacles")
    coordinates = fp.generate_coordinates_archi2()
    print("finish generating coordinates")
    t = time.time()
    communication_graph = sp.generate_list_connections_between_positions(coordinates, obstacles, sensitivity, Rc)
    print("finish generating graph of zones in ", end=" ")
    print(time.time() - t, end=" s")
    return nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles


def init_archi3():
    nb_zones = 20 * 12 + 5 * 2 * 3
    nb_targets = 94
    file1 = open('targets_archi3', 'r')
    list_target_points = [int(x) for x in file1.readline().split(',')]
    obstacles = fp.load_obstacles("archi3")
    print("finish loading obstacles")
    coordinates = fp.generate_coordinates_archi3()
    print("finish generating coordinates")
    t = time.time()
    communication_graph = sp.generate_list_connections_between_positions(coordinates, obstacles, sensitivity, Rc)
    print("finish generating graph of zones in ", end=" ")
    print(time.time() - t, end=" s")
    return nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles


def init_archi4():
    nb_zones = 15 * 12
    nb_targets = 62
    file1 = open('targets_archi4', 'r')
    list_target_points = [int(x) for x in file1.readline().split(',')]
    obstacles = fp.load_obstacles("archi4")
    print("finish loading obstacles")
    coordinates = fp.generate_coordinates_archi4()
    print("finish generating coordinates")
    t = time.time()
    communication_graph = sp.generate_list_connections_between_positions(coordinates, obstacles, sensitivity, Rc)
    print("finish generating graph of zones in ", end=" ")
    print(time.time() - t, end=" s")
    return nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles



def init_archi5():
    nb_zones = 26 * 20
    nb_targets = 182
    file1 = open('targets_archi5', 'r')
    list_target_points = [int(x) for x in file1.readline().split(',')]
    obstacles = fp.load_obstacles("archi5")
    print("finish loading obstacles")
    coordinates = fp.generate_coordinates_archi5()
    print("finish generating coordinates")
    t = time.time()
    communication_graph = sp.generate_list_connections_between_positions(coordinates, obstacles, sensitivity, Rc)
    print("finish generating graph of zones in ", end=" ")
    print(time.time() - t, end=" s")
    return nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles



def init_archi6():
    nb_zones = 22 * 10 + 18 * 6
    nb_targets = 114
    file1 = open('targets_archi6', 'r')
    list_target_points = [int(x) for x in file1.readline().split(',')]
    obstacles = fp.load_obstacles("archi6")
    print("finish loading obstacles")
    coordinates = fp.generate_coordinates_archi6()
    print("finish generating coordinates")
    t = time.time()
    communication_graph = sp.generate_list_connections_between_positions(coordinates, obstacles, sensitivity, Rc)
    print("finish generating graph of zones in ", end=" ")
    print(time.time() - t, end=" s")
    return nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles



def init_archi7():
    nb_zones = 7 * 10 + 15 * 23
    nb_targets = 145
    file1 = open('targets_archi7', 'r')
    list_target_points = [int(x) for x in file1.readline().split(',')]
    obstacles = fp.load_obstacles("archi7")
    #print("finish loading obstacles")
    coordinates = fp.generate_coordinates_archi7()
    #print("finish generating coordinates")
    t = time.time()
    communication_graph = sp.generate_list_connections_between_positions(coordinates, obstacles, sensitivity, Rc)
    #print("finish generating graph of zones in ", end=" ")
    #print(time.time() - t, end=" s")
    return nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles



def init_archi8():
    nb_zones = 2 * 5 * 30 + 25 * 20
    nb_targets = 280
    file1 = open('targets_archi8', 'r')
    list_target_points = [int(x) for x in file1.readline().split(',')]
    obstacles = fp.load_obstacles("archi8")
    #print("finish loading obstacles")
    coordinates = fp.generate_coordinates_archi8()
    #print("finish generating coordinates")
    t = time.time()
    communication_graph = sp.generate_list_connections_between_positions(coordinates, obstacles, sensitivity, Rc)
    #print("finish generating graph of zones in ", end=" ")
    #print(time.time() - t, end=" s")
    return nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles



def init_archi9():
    nb_zones = 10 * 13 + 5 * 25
    nb_targets = 80
    file1 = open('targets_archi9', 'r')
    list_target_points = [int(x) for x in file1.readline().split(',')]
    obstacles = fp.load_obstacles("archi9")
    #print("finish loading obstacles")
    coordinates = fp.generate_coordinates_archi9()
    #print("finish generating coordinates")
    t = time.time()
    communication_graph = sp.generate_list_connections_between_positions(coordinates, obstacles, sensitivity, Rc)
    #print("finish generating graph of zones in ", end=" ")
    #print(time.time() - t, end=" s")
    return nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles



def init_archi10():
    nb_zones = 10 * 20 + 70 * 20
    nb_targets = 560
    file1 = open('targets_archi10', 'r')
    list_target_points = [int(x) for x in file1.readline().split(',')]
    obstacles = fp.load_obstacles("archi10")
    print("finish loading obstacles")
    coordinates = fp.generate_coordinates_archi10()
    print("finish generating coordinates")
    t = time.time()
    communication_graph = sp.generate_list_connections_between_positions(coordinates, obstacles, sensitivity, Rc)
    print("finish generating graph of zones in ", end=" ")
    print(time.time() - t, end=" s")
    return nb_zones, nb_targets, list_target_points, communication_graph,  coordinates, obstacles
