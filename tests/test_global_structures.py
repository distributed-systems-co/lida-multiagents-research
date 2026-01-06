#!/usr/bin/env python3
"""
Test global structures for world-scale hierarchies.

Tests thousands of interconnected organizations with:
- Geographic hierarchies (regions, sub-regions, countries)
- Organization types (governments, corps, NGOs, universities, etc.)
- Tag-based meta-facets (industry, ideology, influence, size, reach, power, alignment)
- Inter-organization relationships (alliances, rivalries)
- Role-based agent population
"""

import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")

import pytest

from src.meta.global_structures import (
    Region,
    SubRegion,
    Country,
    COUNTRIES,
    OrgType,
    HierarchyType,
    IndustryTag,
    IdeologyTag,
    InfluenceTag,
    SizeTag,
    ReachTag,
    PowerTag,
    AlignmentTag,
    RoleTemplate,
    CORPORATE_ROLES,
    GOVERNMENT_ROLES,
    MILITARY_ROLES,
    RELIGIOUS_ROLES,
    ACADEMIC_ROLES,
    NGO_ROLES,
    Organization,
    GlobalAgent,
    GlobalNetwork,
)


@pytest.fixture
def network():
    """Create a GlobalNetwork with sample data for testing."""
    net = GlobalNetwork()
    net.generate_global_corporations(count=20)
    net.generate_ngos(count=10)
    net.generate_universities(count=10)
    return net


def test_geographic_hierarchies():
    """Test geographic hierarchy structures."""
    print("\n" + "=" * 70)
    print("GEOGRAPHIC HIERARCHIES")
    print("=" * 70)

    # Regions
    print(f"\nRegions: {len(Region)} total")
    for region in Region:
        print(f"  - {region.value}")

    # Sub-regions
    print(f"\nSub-Regions: {len(SubRegion)} total")
    by_prefix = defaultdict(list)
    for sr in SubRegion:
        prefix = sr.value.split("_")[0]
        by_prefix[prefix].append(sr.value)
    for prefix, srs in sorted(by_prefix.items()):
        print(f"  {prefix}: {len(srs)} sub-regions")

    # Countries
    print(f"\nSample Countries: {len(COUNTRIES)} defined")
    by_region = defaultdict(list)
    for code, country in COUNTRIES.items():
        by_region[country.region.value].append(f"{code} ({country.name})")
    for region, countries in sorted(by_region.items()):
        print(f"  {region}: {', '.join(countries)}")

    print("\nAlliances in sample countries:")
    all_alliances = set()
    for country in COUNTRIES.values():
        all_alliances.update(country.alliances)
    for alliance in sorted(all_alliances):
        members = [c.name for c in COUNTRIES.values() if alliance in c.alliances]
        print(f"  {alliance}: {len(members)} members - {', '.join(members[:5])}{'...' if len(members) > 5 else ''}")


def test_organization_types():
    """Test organization type enumeration."""
    print("\n" + "=" * 70)
    print("ORGANIZATION TYPES")
    print("=" * 70)

    print(f"\nOrganization Types: {len(OrgType)} total")
    by_category = defaultdict(list)
    for ot in OrgType:
        category = ot.value.split("_")[0]
        by_category[category].append(ot.value)

    for category, types in sorted(by_category.items()):
        print(f"\n  {category.upper()}: {len(types)} types")
        for t in types:
            print(f"    - {t}")

    print(f"\nHierarchy Types: {len(HierarchyType)}")
    for ht in HierarchyType:
        print(f"  - {ht.value}")


def test_tag_facets():
    """Test tag-based meta-facets."""
    print("\n" + "=" * 70)
    print("TAG-BASED META-FACETS")
    print("=" * 70)

    facets = [
        ("Industry", IndustryTag),
        ("Ideology", IdeologyTag),
        ("Influence", InfluenceTag),
        ("Size", SizeTag),
        ("Reach", ReachTag),
        ("Power", PowerTag),
        ("Alignment", AlignmentTag),
    ]

    total_tags = 0
    for name, enum_class in facets:
        count = len(enum_class)
        total_tags += count
        print(f"\n{name}Tag: {count} values")
        values = [e.value for e in enum_class]
        # Print in rows of 5
        for i in range(0, len(values), 5):
            print(f"  {', '.join(values[i:i+5])}")

    print(f"\nTotal tag dimensions: {len(facets)}")
    print(f"Total unique tag values: {total_tags}")


def test_role_templates():
    """Test role template hierarchies."""
    print("\n" + "=" * 70)
    print("ROLE TEMPLATES")
    print("=" * 70)

    templates = [
        ("Corporate", CORPORATE_ROLES),
        ("Government", GOVERNMENT_ROLES),
        ("Military", MILITARY_ROLES),
        ("Religious", RELIGIOUS_ROLES),
        ("Academic", ACADEMIC_ROLES),
        ("NGO", NGO_ROLES),
    ]

    for name, roles in templates:
        print(f"\n{name} Hierarchy: {len(roles)} roles")
        max_level = max(r.level for r in roles)
        for level in range(max_level + 1):
            level_roles = [r for r in roles if r.level == level]
            if level_roles:
                role_names = [r.name for r in level_roles]
                print(f"  Level {level}: {', '.join(role_names)}")


def test_small_network():
    """Test small network generation."""
    print("\n" + "=" * 70)
    print("SMALL NETWORK TEST")
    print("=" * 70)

    network = GlobalNetwork()

    # Create a few organizations manually
    print("\nCreating manual organizations...")

    # Tech company
    tech = network.create_organization(
        name="TechGiant_Inc",
        org_type=OrgType.CORPORATION_MULTINATIONAL,
        headquarters="US",
        operating_countries={"US", "GB", "DE", "JP", "IN"},
        industries={IndustryTag.TECHNOLOGY},
        size=SizeTag.MEGA,
        reach=ReachTag.GLOBAL,
        power=PowerTag.MAJOR_PLAYER,
        alignment=AlignmentTag.WESTERN,
    )
    print(f"  Created: {tech.name} ({tech.org_type.value})")
    print(f"    Tags: {', '.join(list(tech.all_tags())[:10])}...")

    # Government
    govt = network.create_organization(
        name="US_State_Dept",
        org_type=OrgType.GOVERNMENT_AGENCY,
        headquarters="US",
        size=SizeTag.ENTERPRISE,
        reach=ReachTag.GLOBAL,
        power=PowerTag.SUPERPOWER,
        influences={InfluenceTag.DIPLOMATIC, InfluenceTag.POLITICAL},
    )
    print(f"  Created: {govt.name} ({govt.org_type.value})")

    # NGO
    ngo = network.create_organization(
        name="Global_Climate_Action",
        org_type=OrgType.NGO_INTERNATIONAL,
        headquarters="CH",
        size=SizeTag.LARGE,
        reach=ReachTag.GLOBAL,
        ideologies={IdeologyTag.ENVIRONMENTALIST, IdeologyTag.PROGRESSIVE},
        influences={InfluenceTag.SOCIAL, InfluenceTag.CULTURAL},
    )
    print(f"  Created: {ngo.name} ({ngo.org_type.value})")

    # Add relationships
    network.add_alliance(govt.id, tech.id)
    network.add_rivalry(tech.id, ngo.id)

    print("\nRelationships:")
    print(f"  Alliances: {govt.name} <-> {tech.name}")
    print(f"  Rivalries: {tech.name} <-> {ngo.name}")

    # Populate with agents
    print("\nPopulating organizations with agents...")
    tech_agents = network.populate_organization(tech.id, CORPORATE_ROLES, scale=0.5)
    print(f"  {tech.name}: {len(tech_agents)} agents")

    govt_agents = network.populate_organization(govt.id, GOVERNMENT_ROLES, scale=0.3)
    print(f"  {govt.name}: {len(govt_agents)} agents")

    ngo_agents = network.populate_organization(ngo.id, NGO_ROLES, scale=0.4)
    print(f"  {ngo.name}: {len(ngo_agents)} agents")

    # Test lookups
    print("\nIndex lookups:")
    us_orgs = network.get_orgs_by_country("US")
    print(f"  Organizations in US: {len(us_orgs)}")

    tech_orgs = network.get_orgs_by_tag("technology")
    print(f"  Organizations with 'technology' tag: {len(tech_orgs)}")

    global_orgs = network.get_orgs_by_tag("global")
    print(f"  Organizations with 'global' reach: {len(global_orgs)}")

    stats = network.get_stats()
    print(f"\nNetwork stats:")
    print(f"  Total organizations: {stats['total_organizations']}")
    print(f"  Total agents: {stats['total_agents']}")
    print(f"  Countries covered: {stats['countries_covered']}")


def test_country_government_generation():
    """Test government structure generation for countries."""
    print("\n" + "=" * 70)
    print("COUNTRY GOVERNMENT GENERATION")
    print("=" * 70)

    network = GlobalNetwork()

    test_countries = ["US", "CN", "DE", "JP", "BR"]

    for country_code in test_countries:
        country = COUNTRIES.get(country_code)
        print(f"\nGenerating government for {country.name}...")

        orgs = network.generate_country_government(country_code)
        print(f"  Created {len(orgs)} organizations:")
        for org in orgs:
            print(f"    - {org.name} ({org.org_type.value})")

    stats = network.get_stats()
    print(f"\nTotal government organizations: {stats['total_organizations']}")

    # Show by type
    print("\nOrganizations by type:")
    for org_type, count in sorted(stats['org_types'].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {org_type}: {count}")


def test_bulk_generation():
    """Test bulk organization generation."""
    print("\n" + "=" * 70)
    print("BULK GENERATION TEST")
    print("=" * 70)

    network = GlobalNetwork()

    # Generate corporations
    print("\nGenerating global corporations...")
    start = time.time()
    corps = network.generate_global_corporations(count=100)
    corp_time = time.time() - start
    print(f"  Created {len(corps)} corporations in {corp_time:.3f}s")

    # Generate NGOs
    print("\nGenerating NGOs...")
    start = time.time()
    ngos = network.generate_ngos(count=50)
    ngo_time = time.time() - start
    print(f"  Created {len(ngos)} NGOs in {ngo_time:.3f}s")

    # Generate universities
    print("\nGenerating universities...")
    start = time.time()
    unis = network.generate_universities(count=60)
    uni_time = time.time() - start
    print(f"  Created {len(unis)} universities in {uni_time:.3f}s")

    # Industry distribution
    stats = network.get_stats()
    print("\nIndustry distribution:")
    for industry, count in sorted(stats['industries'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {industry}: {count}")

    # Country distribution
    print("\nCountry distribution (top 10):")
    country_counts = defaultdict(int)
    for org in network.organizations.values():
        country_counts[org.headquarters_country] += 1
    for country, count in sorted(country_counts.items(), key=lambda x: -x[1])[:10]:
        country_name = COUNTRIES[country].name if country in COUNTRIES else country
        print(f"  {country_name}: {count}")


def test_full_global_network():
    """Test full global network generation at scale."""
    print("\n" + "=" * 70)
    print("FULL GLOBAL NETWORK GENERATION")
    print("=" * 70)

    network = GlobalNetwork()

    print("\nGenerating full global network...")
    print("  - 10 country governments")
    print("  - 100 multinational corporations")
    print("  - 50 international NGOs")
    print("  - 60 universities")
    print("  - Populating key organizations with agents")

    start = time.time()
    stats = network.generate_full_global_network(
        govt_countries=["US", "CN", "DE", "GB", "FR", "JP", "IN", "BR", "RU", "AU"],
        corps=100,
        ngos=50,
        universities=60,
    )
    gen_time = time.time() - start

    print(f"\nGeneration completed in {gen_time:.2f}s")
    print(f"\nGeneration stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Full network stats
    full_stats = network.get_stats()
    print(f"\nFull network statistics:")
    print(f"  Total organizations: {full_stats['total_organizations']}")
    print(f"  Total agents: {full_stats['total_agents']}")
    print(f"  Countries covered: {full_stats['countries_covered']}")
    print(f"  Regions covered: {full_stats['regions_covered']}")

    print(f"\nOrganization types (top 10):")
    for org_type, count in sorted(full_stats['org_types'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {org_type}: {count}")

    print(f"\nAlignment distribution:")
    for alignment, count in sorted(full_stats['alignments'].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {alignment}: {count}")

    return network


def test_tag_based_queries(network: GlobalNetwork):
    """Test tag-based filtering and queries."""
    print("\n" + "=" * 70)
    print("TAG-BASED QUERIES")
    print("=" * 70)

    # Single tag queries
    print("\nSingle tag queries:")

    tech_orgs = network.get_orgs_by_tag("technology")
    print(f"  'technology' tag: {len(tech_orgs)} organizations")

    global_orgs = network.get_orgs_by_tag("global")
    print(f"  'global' reach: {len(global_orgs)} organizations")

    mega_orgs = network.get_orgs_by_tag("mega")
    print(f"  'mega' size: {len(mega_orgs)} organizations")

    western_orgs = network.get_orgs_by_tag("western")
    print(f"  'western' alignment: {len(western_orgs)} organizations")

    # Multi-tag queries
    print("\nMulti-tag queries:")

    # Western + Global (any)
    west_global = network.get_orgs_by_tags({"western", "global"}, match_all=False)
    print(f"  'western' OR 'global': {len(west_global)} organizations")

    # Tech + Global (all)
    tech_global = network.get_orgs_by_tags({"technology", "global"}, match_all=True)
    print(f"  'technology' AND 'global': {len(tech_global)} organizations")

    # Major power tech companies
    major_tech = network.get_orgs_by_tags({"technology", "major_player"}, match_all=True)
    print(f"  'technology' AND 'major_player': {len(major_tech)} organizations")

    # Country-based queries
    print("\nCountry-based queries:")
    for country in ["US", "CN", "DE", "JP"]:
        orgs = network.get_orgs_by_country(country)
        agents = network.get_agents_by_country(country)
        country_name = COUNTRIES[country].name
        print(f"  {country_name}: {len(orgs)} orgs, {len(agents)} agents")

    # Region-based queries
    print("\nRegion-based queries:")
    for region in [Region.NORTH_AMERICA, Region.EUROPE, Region.EAST_ASIA]:
        orgs = network.get_orgs_by_region(region)
        print(f"  {region.value}: {len(orgs)} organizations")


def test_scaled_network_sizes():
    """Test network generation at different scales."""
    print("\n" + "=" * 70)
    print("SCALED NETWORK SIZES")
    print("=" * 70)

    scales = [
        {"govts": 5, "corps": 50, "ngos": 20, "unis": 20},
        {"govts": 10, "corps": 100, "ngos": 50, "unis": 40},
        {"govts": 15, "corps": 200, "ngos": 100, "unis": 80},
        {"govts": 20, "corps": 500, "ngos": 200, "unis": 150},
    ]

    govt_countries = list(COUNTRIES.keys())

    for i, scale in enumerate(scales):
        network = GlobalNetwork()

        print(f"\nScale {i+1}: {scale['govts']} govts, {scale['corps']} corps, {scale['ngos']} NGOs, {scale['unis']} unis")

        start = time.time()
        stats = network.generate_full_global_network(
            govt_countries=govt_countries[:scale['govts']],
            corps=scale['corps'],
            ngos=scale['ngos'],
            universities=scale['unis'],
        )
        gen_time = time.time() - start

        full_stats = network.get_stats()
        print(f"  Generated in {gen_time:.2f}s")
        print(f"  Total orgs: {full_stats['total_organizations']}")
        print(f"  Total agents: {full_stats['total_agents']}")
        print(f"  Orgs/second: {full_stats['total_organizations']/gen_time:.0f}")


def test_cross_organization_relationships():
    """Test inter-organization relationships."""
    print("\n" + "=" * 70)
    print("CROSS-ORGANIZATION RELATIONSHIPS")
    print("=" * 70)

    network = GlobalNetwork()

    # Create competing tech companies
    companies = []
    for name in ["TechAlpha", "TechBeta", "TechGamma", "TechDelta"]:
        org = network.create_organization(
            name=name,
            org_type=OrgType.CORPORATION_MULTINATIONAL,
            headquarters="US",
            industries={IndustryTag.TECHNOLOGY},
            size=SizeTag.MEGA,
            reach=ReachTag.GLOBAL,
        )
        companies.append(org)

    # Create alliances
    network.add_alliance(companies[0].id, companies[1].id)  # Alpha + Beta
    network.add_alliance(companies[2].id, companies[3].id)  # Gamma + Delta

    # Create rivalries between alliances
    network.add_rivalry(companies[0].id, companies[2].id)  # Alpha vs Gamma
    network.add_rivalry(companies[0].id, companies[3].id)  # Alpha vs Delta
    network.add_rivalry(companies[1].id, companies[2].id)  # Beta vs Gamma
    network.add_rivalry(companies[1].id, companies[3].id)  # Beta vs Delta

    print("\nAlliance structure:")
    for org in companies:
        allies = network.get_allied_orgs(org.id)
        rivals = network.get_rival_orgs(org.id)
        print(f"  {org.name}:")
        print(f"    Allies: {[a.name for a in allies]}")
        print(f"    Rivals: {[r.name for r in rivals]}")

    # Test cross-org agent connections
    print("\nPopulating organizations...")
    all_agents = []
    for org in companies:
        agents = network.populate_organization(org.id, CORPORATE_ROLES, scale=0.3)
        all_agents.extend(agents)
        print(f"  {org.name}: {len(agents)} agents")

    # Create cross-org connections (e.g., board members, advisors)
    print("\nCreating cross-organization connections...")
    import random

    # Get senior agents (level 0-2) from each org
    senior_agents = [a for a in all_agents if a.level <= 2]

    # Connect seniors across allied organizations
    connection_count = 0
    for agent in senior_agents:
        agent_org = network.organizations[agent.organization_id]
        allied_orgs = network.get_allied_orgs(agent_org.id)

        for allied_org in allied_orgs:
            allied_seniors = [a for a in network.get_agents_by_org(allied_org.id) if a.level <= 2]
            if allied_seniors:
                # Connect to one random senior
                target = random.choice(allied_seniors)
                agent.cross_org_connections.add(target.id)
                target.cross_org_connections.add(agent.id)
                connection_count += 1

    print(f"  Created {connection_count} cross-organization connections")

    # Show sample connections
    print("\nSample cross-org connections:")
    for agent in senior_agents[:5]:
        if agent.cross_org_connections:
            connections = [network.agents[aid].name for aid in agent.cross_org_connections]
            print(f"  {agent.name} <-> {', '.join(connections)}")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("GLOBAL STRUCTURES TEST SUITE")
    print("Testing world-scale hierarchies with thousands of organizations")
    print("=" * 70)

    # Basic structure tests
    test_geographic_hierarchies()
    test_organization_types()
    test_tag_facets()
    test_role_templates()

    # Network generation tests
    test_small_network()
    test_country_government_generation()
    test_bulk_generation()

    # Full network test
    network = test_full_global_network()

    # Query tests
    test_tag_based_queries(network)

    # Relationship tests
    test_cross_organization_relationships()

    # Scale tests
    test_scaled_network_sizes()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
