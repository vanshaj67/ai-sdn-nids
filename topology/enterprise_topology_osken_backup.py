#!/usr/bin/env python3

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController


def enterprise_topology():
    net = Mininet(
        controller=RemoteController,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        build=False,
    )

    net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)
    # Core and zone switches
    core = net.addSwitch("s1", failMode="secure", protocols="OpenFlow13")
    hr_sw = net.addSwitch("s2", failMode="secure", protocols="OpenFlow13")
    finance_sw = net.addSwitch("s3", failMode="secure", protocols="OpenFlow13")
    it_sw = net.addSwitch("s4", failMode="secure", protocols="OpenFlow13")
    iot_sw = net.addSwitch("s5", failMode="secure", protocols="OpenFlow13")
    dmz_sw = net.addSwitch("s6", failMode="secure", protocols="OpenFlow13")

    # Core-to-zone links
    for zone_sw in [hr_sw, finance_sw, it_sw, iot_sw, dmz_sw]:
        net.addLink(core, zone_sw, bw=100)

    # HR department
    hr1 = net.addHost("hr1", ip="10.0.10.11/16")
    hr2 = net.addHost("hr2", ip="10.0.10.12/16")
    net.addLink(hr1, hr_sw, bw=20)
    net.addLink(hr2, hr_sw, bw=20)

    # Finance department
    fin1 = net.addHost("fin1", ip="10.0.20.11/16")
    fin2 = net.addHost("fin2", ip="10.0.20.12/16")
    net.addLink(fin1, finance_sw, bw=20)
    net.addLink(fin2, finance_sw, bw=20)

    # IT department
    it1 = net.addHost("it1", ip="10.0.30.11/16")
    it2 = net.addHost("it2", ip="10.0.30.12/16")
    atkint = net.addHost("atkint", ip="10.0.90.20/16")
    net.addLink(it1, it_sw, bw=20)
    net.addLink(it2, it_sw, bw=20)
    net.addLink(atkint, it_sw, bw=20)

    # IoT zone
    camera = net.addHost("camera", ip="10.0.40.11/16")
    sensor = net.addHost("sensor", ip="10.0.40.12/16")
    doorlock = net.addHost("doorlock", ip="10.0.40.13/16")
    rpi = net.addHost("rpi", ip="10.0.40.14/16")
    for device in [camera, sensor, doorlock, rpi]:
        net.addLink(device, iot_sw, bw=10)

    # DMZ servers
    web = net.addHost("web", ip="10.0.50.11/16")
    dns = net.addHost("dns", ip="10.0.50.12/16")
    ftp = net.addHost("ftp", ip="10.0.50.13/16")
    database = net.addHost("database", ip="10.0.50.14/16")
    for server in [web, dns, ftp, database]:
        net.addLink(server, dmz_sw, bw=50)

    # External attacker
    atkext = net.addHost("atkext", ip="10.0.90.10/16")
    net.addLink(atkext, core, bw=20)

    net.build()
    net.start()

    info("\n*** AI-Driven SDN Enterprise Testbed is running\n")
    info("*** Type 'nodes', 'net', and 'pingall' to test it.\n")
    info("*** Type 'exit' when finished.\n\n")
    CLI(net)

    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    enterprise_topology()
