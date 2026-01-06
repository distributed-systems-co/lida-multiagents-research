#!/usr/bin/env python3
"""
Test GDELT integration and world datasets.

Tests:
- GDELT data fetching (events, mentions, GKG)
- Cron job scheduling
- World datasets (leaders, corporations, orgs)
- Data aggregation and queries
"""

import sys
import asyncio
import time
from collections import defaultdict

sys.path.insert(0, ".")

from src.meta.gdelt_integration import (
    GDELTEventCode,
    GDELTActorType,
    GDELTActor,
    GDELTEvent,
    GDELTMention,
    GKGRecord,
    GDELTFetcher,
    CronJob,
    CronScheduler,
    GDELTDataManager,
)
from src.meta.world_datasets import (
    WORLD_LEADERS,
    MAJOR_CORPORATIONS,
    INTERNATIONAL_ORGS,
    CENTRAL_BANKS,
    THINK_TANKS,
    MEDIA_ORGANIZATIONS,
    TOP_UNIVERSITIES,
    RELIGIOUS_ORGANIZATIONS,
    get_all_entities,
    get_entities_by_country,
    get_entities_by_tag,
    get_dataset_stats,
)


def test_world_leaders():
    """Test world leaders dataset."""
    print("\n" + "=" * 70)
    print("WORLD LEADERS DATASET")
    print("=" * 70)

    print(f"\nTotal world leaders: {len(WORLD_LEADERS)}")

    # Group by ideology
    by_ideology = defaultdict(list)
    for key, leader in WORLD_LEADERS.items():
        by_ideology[leader.ideology].append(leader.name)

    print("\nBy ideology:")
    for ideology, leaders in sorted(by_ideology.items()):
        print(f"  {ideology}: {len(leaders)}")
        for name in leaders[:3]:
            print(f"    - {name}")
        if len(leaders) > 3:
            print(f"    ... and {len(leaders) - 3} more")

    # G7 leaders
    print("\nG7 Leaders:")
    g7_countries = ["US", "GB", "FR", "DE", "IT", "JP", "CA"]
    for key, leader in WORLD_LEADERS.items():
        if leader.country_code in g7_countries:
            print(f"  {leader.country_code}: {leader.name} ({leader.title})")


def test_corporations():
    """Test corporations dataset."""
    print("\n" + "=" * 70)
    print("MAJOR CORPORATIONS DATASET")
    print("=" * 70)

    print(f"\nTotal corporations: {len(MAJOR_CORPORATIONS)}")

    # Group by industry
    by_industry = defaultdict(list)
    for ticker, corp in MAJOR_CORPORATIONS.items():
        by_industry[corp.industry].append((ticker, corp.name, corp.market_cap_billions))

    print("\nBy industry:")
    for industry, corps in sorted(by_industry.items()):
        total_cap = sum(c[2] for c in corps)
        print(f"\n  {industry.upper()} ({len(corps)} companies, ${total_cap:.0f}B total market cap):")
        for ticker, name, cap in sorted(corps, key=lambda x: -x[2])[:5]:
            print(f"    {ticker}: {name} (${cap:.0f}B)")

    # Top 10 by market cap
    print("\nTop 10 by Market Cap:")
    sorted_corps = sorted(MAJOR_CORPORATIONS.items(), key=lambda x: -x[1].market_cap_billions)
    for i, (ticker, corp) in enumerate(sorted_corps[:10], 1):
        print(f"  {i}. {corp.name} ({ticker}): ${corp.market_cap_billions:.0f}B")


def test_international_orgs():
    """Test international organizations dataset."""
    print("\n" + "=" * 70)
    print("INTERNATIONAL ORGANIZATIONS DATASET")
    print("=" * 70)

    print(f"\nTotal organizations: {len(INTERNATIONAL_ORGS)}")

    # Group by type
    by_type = defaultdict(list)
    for abbrev, org in INTERNATIONAL_ORGS.items():
        by_type[org.org_type].append(org)

    print("\nBy type:")
    for org_type, orgs in sorted(by_type.items()):
        print(f"\n  {org_type.upper()}:")
        for org in sorted(orgs, key=lambda x: -x.member_count):
            print(f"    {org.abbreviation}: {org.name} ({org.member_count} members)")

    # By budget
    print("\nTop 10 by Budget:")
    sorted_orgs = sorted(INTERNATIONAL_ORGS.items(), key=lambda x: -x[1].budget_billions)
    for i, (abbrev, org) in enumerate(sorted_orgs[:10], 1):
        if org.budget_billions > 0:
            print(f"  {i}. {org.name}: ${org.budget_billions:.1f}B")


def test_central_banks():
    """Test central banks dataset."""
    print("\n" + "=" * 70)
    print("CENTRAL BANKS DATASET")
    print("=" * 70)

    print(f"\nTotal central banks: {len(CENTRAL_BANKS)}")

    # By reserves
    print("\nBy Foreign Reserves:")
    sorted_banks = sorted(CENTRAL_BANKS.items(), key=lambda x: -x[1].reserves_billions)
    for abbrev, bank in sorted_banks:
        print(f"  {abbrev}: {bank.name} - ${bank.reserves_billions}B ({bank.currency})")
        print(f"    Governor: {bank.governor}")


def test_think_tanks():
    """Test think tanks dataset."""
    print("\n" + "=" * 70)
    print("THINK TANKS DATASET")
    print("=" * 70)

    print(f"\nTotal think tanks: {len(THINK_TANKS)}")

    # Group by ideology
    by_ideology = defaultdict(list)
    for key, tt in THINK_TANKS.items():
        by_ideology[tt.ideology].append(tt)

    print("\nBy ideology:")
    for ideology, tanks in sorted(by_ideology.items()):
        print(f"\n  {ideology.upper()}:")
        for tt in tanks:
            print(f"    {tt.abbreviation}: {tt.name} ({tt.headquarters_country})")

    # Focus areas
    print("\nMost common focus areas:")
    focus_counts = defaultdict(int)
    for tt in THINK_TANKS.values():
        for focus in tt.focus:
            focus_counts[focus] += 1
    for focus, count in sorted(focus_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {focus}: {count}")


def test_media_organizations():
    """Test media organizations dataset."""
    print("\n" + "=" * 70)
    print("MEDIA ORGANIZATIONS DATASET")
    print("=" * 70)

    print(f"\nTotal media organizations: {len(MEDIA_ORGANIZATIONS)}")

    # By type
    by_type = defaultdict(list)
    for key, org in MEDIA_ORGANIZATIONS.items():
        by_type[org.media_type].append(org)

    print("\nBy media type:")
    for media_type, orgs in sorted(by_type.items()):
        print(f"\n  {media_type.upper()}:")
        for org in orgs:
            reach = f"[{org.reach}]" if org.reach else ""
            ideology = f"({org.ideology})" if org.ideology else ""
            print(f"    {org.name} {reach} {ideology}")

    # Global reach
    print("\nGlobal reach organizations:")
    global_orgs = [org for org in MEDIA_ORGANIZATIONS.values() if org.reach == "global"]
    for org in global_orgs:
        print(f"  {org.name} ({org.headquarters_country}) - {org.media_type}")


def test_universities():
    """Test universities dataset."""
    print("\n" + "=" * 70)
    print("TOP UNIVERSITIES DATASET")
    print("=" * 70)

    print(f"\nTotal universities: {len(TOP_UNIVERSITIES)}")

    # By country
    by_country = defaultdict(list)
    for key, uni in TOP_UNIVERSITIES.items():
        by_country[uni.country_code].append(uni)

    print("\nBy country:")
    for country, unis in sorted(by_country.items(), key=lambda x: -len(x[1])):
        print(f"\n  {country} ({len(unis)} universities):")
        for uni in sorted(unis, key=lambda x: -x.endowment_billions)[:5]:
            endowment = f"${uni.endowment_billions:.1f}B" if uni.endowment_billions else ""
            print(f"    {uni.name} ({uni.city}) {endowment}")

    # Top 10 by endowment
    print("\nTop 10 by Endowment:")
    sorted_unis = sorted(TOP_UNIVERSITIES.items(), key=lambda x: -x[1].endowment_billions)
    for i, (key, uni) in enumerate(sorted_unis[:10], 1):
        print(f"  {i}. {uni.name}: ${uni.endowment_billions:.1f}B")


def test_religious_organizations():
    """Test religious organizations dataset."""
    print("\n" + "=" * 70)
    print("RELIGIOUS ORGANIZATIONS DATASET")
    print("=" * 70)

    print(f"\nTotal religious organizations: {len(RELIGIOUS_ORGANIZATIONS)}")

    # By religion
    by_religion = defaultdict(list)
    for key, org in RELIGIOUS_ORGANIZATIONS.items():
        by_religion[org.religion].append(org)

    print("\nBy religion:")
    for religion, orgs in sorted(by_religion.items()):
        print(f"\n  {religion.upper()}:")
        for org in sorted(orgs, key=lambda x: -x.followers_millions):
            print(f"    {org.name} - {org.followers_millions:.0f}M followers")
            print(f"      Leader: {org.leader} ({org.leader_title})")


def test_aggregated_queries():
    """Test aggregated data queries."""
    print("\n" + "=" * 70)
    print("AGGREGATED DATA QUERIES")
    print("=" * 70)

    # Dataset stats
    stats = get_dataset_stats()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Query by country
    print("\nUS Entities:")
    us_entities = get_entities_by_country("US")
    for category, entities in us_entities.items():
        if entities:
            print(f"  {category}: {len(entities)}")

    print("\nChina Entities:")
    cn_entities = get_entities_by_country("CN")
    for category, entities in cn_entities.items():
        if entities:
            print(f"  {category}: {len(entities)}")

    # Query by tag
    print("\nEntities with 'tech' tag:")
    tech_entities = get_entities_by_tag("tech")
    for category, entities in tech_entities.items():
        if entities:
            print(f"  {category}: {len(entities)}")
            for e in entities[:3]:
                print(f"    - {e.name if hasattr(e, 'name') else e}")

    print("\nEntities with 'elite' tag:")
    elite_entities = get_entities_by_tag("elite")
    for category, entities in elite_entities.items():
        if entities:
            print(f"  {category}: {len(entities)}")


def test_gdelt_structures():
    """Test GDELT data structures."""
    print("\n" + "=" * 70)
    print("GDELT DATA STRUCTURES")
    print("=" * 70)

    # Event codes
    print(f"\nGDELT Event Codes: {len(GDELTEventCode)}")
    for code in GDELTEventCode:
        print(f"  {code.value}: {code.name}")

    # Actor types
    print(f"\nGDELT Actor Types: {len(GDELTActorType)}")
    for actor_type in GDELTActorType:
        print(f"  {actor_type.value}: {actor_type.name}")

    # Test event creation
    print("\nSample Event:")
    event = GDELTEvent(
        global_event_id="test123",
        day=20241215,
        actor1=GDELTActor(
            code="USA",
            name="United States",
            country_code="US",
            type_codes=["GOV"],
        ),
        actor2=GDELTActor(
            code="CHN",
            name="China",
            country_code="CN",
            type_codes=["GOV"],
        ),
        event_code="040",
        quad_class=1,
        goldstein_scale=3.0,
        avg_tone=-1.5,
    )
    print(f"  Event ID: {event.global_event_id}")
    print(f"  Actor1: {event.actor1.name}")
    print(f"  Actor2: {event.actor2.name}")
    print(f"  Cooperative: {event.is_cooperative()}")
    print(f"  Conflictual: {event.is_conflictual()}")
    print(f"  Goldstein: {event.goldstein_scale}")


async def test_cron_scheduler():
    """Test cron job scheduler."""
    print("\n" + "=" * 70)
    print("CRON JOB SCHEDULER")
    print("=" * 70)

    scheduler = CronScheduler()

    # Track execution
    execution_log = []

    def log_execution(name: str):
        execution_log.append((time.time(), name))
        print(f"  Executed: {name}")

    # Add test jobs
    scheduler.add_job("every_2s", "every 2s", lambda: log_execution("every_2s"))
    scheduler.add_job("every_3s", "every 3s", lambda: log_execution("every_3s"))
    scheduler.add_job("once", "once", lambda: log_execution("once"))

    print("\nScheduler status (before start):")
    status = scheduler.get_status()
    print(f"  Running: {status['running']}")
    print(f"  Jobs: {len(status['jobs'])}")
    for name, job_status in status['jobs'].items():
        print(f"    {name}: enabled={job_status['enabled']}, schedule={job_status['schedule']}")

    print("\nStarting scheduler for 5 seconds...")
    scheduler.start()

    # Run for 5 seconds
    await asyncio.sleep(5)

    scheduler.stop()

    print("\nScheduler status (after stop):")
    status = scheduler.get_status()
    print(f"  Running: {status['running']}")
    for name, job_status in status['jobs'].items():
        print(f"    {name}: run_count={job_status['run_count']}, errors={job_status['error_count']}")

    print(f"\nTotal executions logged: {len(execution_log)}")


async def test_gdelt_fetcher():
    """Test GDELT data fetcher (requires network)."""
    print("\n" + "=" * 70)
    print("GDELT DATA FETCHER (Network Test)")
    print("=" * 70)

    print("\nAttempting to fetch latest GDELT files...")
    print("(This requires network access to GDELT servers)")

    try:
        async with GDELTFetcher() as fetcher:
            # Get latest file URLs
            files = await fetcher.get_latest_files()
            print(f"\nLatest files available:")
            for file_type, url in files.items():
                print(f"  {file_type}: {url[:80]}...")

            # Fetch events (limited)
            print("\nFetching events...")
            events = await fetcher.fetch_events()
            print(f"  Fetched {len(events)} events")

            if events:
                print("\nSample events:")
                for event in events[:5]:
                    actor1 = event.actor1.name or event.actor1.country_code or "Unknown"
                    actor2 = event.actor2.name or event.actor2.country_code or "Unknown"
                    print(f"  [{event.day}] {actor1} -> {actor2}")
                    print(f"    Event: {event.event_code}, Tone: {event.avg_tone:.1f}")

                # Statistics
                print("\nEvent statistics:")
                cooperative = sum(1 for e in events if e.is_cooperative())
                conflictual = sum(1 for e in events if e.is_conflictual())
                print(f"  Cooperative: {cooperative} ({100*cooperative/len(events):.1f}%)")
                print(f"  Conflictual: {conflictual} ({100*conflictual/len(events):.1f}%)")

                # Country distribution
                countries = defaultdict(int)
                for e in events:
                    if e.actor1.country_code:
                        countries[e.actor1.country_code] += 1
                    if e.actor2.country_code:
                        countries[e.actor2.country_code] += 1

                print("\nTop 10 countries by event involvement:")
                for country, count in sorted(countries.items(), key=lambda x: -x[1])[:10]:
                    print(f"  {country}: {count}")

    except Exception as e:
        print(f"\nNetwork error (expected if offline): {e}")
        print("GDELT fetching requires network access to http://data.gdeltproject.org")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("GDELT INTEGRATION & WORLD DATASETS TEST SUITE")
    print("=" * 70)

    # World datasets tests
    test_world_leaders()
    test_corporations()
    test_international_orgs()
    test_central_banks()
    test_think_tanks()
    test_media_organizations()
    test_universities()
    test_religious_organizations()
    test_aggregated_queries()

    # GDELT tests
    test_gdelt_structures()
    asyncio.run(test_cron_scheduler())

    # Network test (optional)
    print("\n" + "=" * 70)
    print("NETWORK TESTS")
    print("=" * 70)
    print("\nRunning GDELT network fetch test...")
    asyncio.run(test_gdelt_fetcher())

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
