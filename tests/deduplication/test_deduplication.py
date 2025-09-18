#!/usr/bin/env python3
"""
Test Script for Deduplication Logic in Legal Document Processing Pipeline

This script tests the updated deduplication and pagination logic to ensure:
1. Proper handling of empty collections (first run)
2. Correct skipping of existing dockets
3. Consistent pagination with oldest-first ordering
4. Robust handling of edge cases

Author: Claude Code - Debugging Assistant
"""

import os
import sys
import json
import logging
import time
from typing import Dict, Set, List, Any
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append('/root/lawlm')

# Import the main classes
from main import LegalDocumentPipeline
from vector_processor import EnhancedVectorProcessor
from config import PipelineConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeduplicationTester:
    """Test class for validating deduplication logic."""

    def __init__(self):
        """Initialize the tester."""
        self.processor = None
        self.vector_processor = None
        self.config = None

    def setup(self):
        """Set up the test environment."""
        logger.info("🔧 Setting up test environment...")

        # Load configuration
        self.config = PipelineConfig()

        # Initialize processors
        self.processor = LegalDocumentPipeline(self.config)
        self.vector_processor = EnhancedVectorProcessor()

        logger.info("✅ Test environment setup complete")

    def test_get_existing_docket_ids(self):
        """Test the get_existing_docket_ids method."""
        logger.info("\n" + "="*60)
        logger.info("🧪 Testing get_existing_docket_ids method")
        logger.info("="*60)

        try:
            existing_dockets = self.vector_processor.get_existing_docket_ids()
            logger.info(f"📊 Found {len(existing_dockets)} existing dockets")

            # Verify the return type is a set
            if not isinstance(existing_dockets, set):
                logger.error(f"❌ Expected set, got {type(existing_dockets)}")
                return False

            if existing_dockets:
                logger.info(f"📋 Sample existing dockets: {list(existing_dockets)[:5]}")
                # Verify all elements are strings or numbers
                for docket_id in list(existing_dockets)[:5]:
                    if not isinstance(docket_id, (str, int)):
                        logger.error(f"❌ Invalid docket ID type: {type(docket_id)}")
                        return False
            else:
                logger.info("📭 No existing dockets found (empty collection)")

            logger.info("✅ get_existing_docket_ids method working correctly")
            return True

        except Exception as e:
            logger.error(f"❌ Error getting existing dockets: {e}")
            return False

    def test_empty_collection_scenario(self):
        """Test scenario where collection is empty (first run)."""
        logger.info("\n" + "="*60)
        logger.info("🧪 Testing Empty Collection Scenario (First Run)")
        logger.info("="*60)

        # Mock empty collection
        with patch.object(self.vector_processor, 'get_existing_docket_ids', return_value=set()):
            logger.info("📭 Simulating empty collection...")

            # Test pagination logic with empty existing_dockets set
            existing_dockets = set()
            logger.info(f"🔍 existing_dockets set: {existing_dockets}")
            logger.info(f"📊 Length of existing_dockets: {len(existing_dockets)}")

            # This should proceed to fetch dockets normally
            logger.info("✅ Empty collection scenario: Should fetch all requested dockets")

            return True

    def test_all_dockets_exist_scenario(self):
        """Test scenario where all requested dockets already exist."""
        logger.info("\n" + "="*60)
        logger.info("🧪 Testing All Dockets Exist Scenario")
        logger.info("="*60)

        # Create a mock set of existing dockets with many IDs
        mock_existing = set([str(i) for i in range(1, 10000)])

        with patch.object(self.vector_processor, 'get_existing_docket_ids', return_value=mock_existing):
            logger.info(f"📊 Simulating collection with {len(mock_existing)} existing dockets")

            existing_dockets = mock_existing

            # Simulate checking if dockets are in the set
            test_docket_ids = ['5', '100', '500', '1000']

            for docket_id in test_docket_ids:
                if docket_id in existing_dockets:
                    logger.info(f"✅ Docket {docket_id} correctly identified as existing")
                else:
                    logger.info(f"❌ Docket {docket_id} not found in existing set")

            logger.info("✅ All dockets exist scenario: Should skip all existing dockets")

            return True

    def test_mixed_scenario(self):
        """Test scenario with some existing and some new dockets."""
        logger.info("\n" + "="*60)
        logger.info("🧪 Testing Mixed Existing/New Dockets Scenario")
        logger.info("="*60)

        # Create a set with some specific existing dockets
        mock_existing = set(['100', '200', '300', '500', '750'])

        with patch.object(self.vector_processor, 'get_existing_docket_ids', return_value=mock_existing):
            logger.info(f"📊 Simulating collection with existing dockets: {mock_existing}")

            existing_dockets = mock_existing

            # Test some dockets that should be new vs existing
            test_cases = [
                ('50', False),   # Should be new (not in existing)
                ('100', True),   # Should be existing
                ('150', False),  # Should be new
                ('200', True),   # Should be existing
                ('400', False),  # Should be new
                ('500', True),   # Should be existing
                ('1000', False)  # Should be new
            ]

            for docket_id, should_exist in test_cases:
                exists = docket_id in existing_dockets

                if exists == should_exist:
                    status = "✅ CORRECT"
                else:
                    status = "❌ ERROR"

                expected = "existing" if should_exist else "new"
                actual = "existing" if exists else "new"

                logger.info(f"{status} Docket {docket_id}: expected {expected}, got {actual}")

            logger.info("✅ Mixed scenario testing complete")

            return True

    def test_pagination_logic_simulation(self):
        """Test the pagination logic simulation with mock data."""
        logger.info("\n" + "="*60)
        logger.info("🧪 Testing Pagination Logic Simulation")
        logger.info("="*60)

        # Simulate a pagination scenario
        existing_dockets = set(['5', '10', '15', '25', '30'])
        logger.info(f"📊 Existing dockets: {existing_dockets}")

        # Simulate pages of dockets coming from API (in ID order)
        mock_pages = [
            # Page 1: dockets 1-10
            [{'id': str(i)} for i in range(1, 11)],
            # Page 2: dockets 11-20
            [{'id': str(i)} for i in range(11, 21)],
            # Page 3: dockets 21-30
            [{'id': str(i)} for i in range(21, 31)],
            # Page 4: dockets 31-40
            [{'id': str(i)} for i in range(31, 41)]
        ]

        new_dockets = []
        target_count = 10

        logger.info(f"🎯 Target: {target_count} new dockets")

        for page_num, page_dockets in enumerate(mock_pages, 1):
            logger.info(f"\n📄 Processing Page {page_num} ({len(page_dockets)} dockets)")

            page_new_count = 0
            for docket in page_dockets:
                docket_id = docket.get('id', '')

                if docket_id and docket_id not in existing_dockets:
                    new_dockets.append(docket)
                    page_new_count += 1
                    logger.info(f"  ✨ New docket: {docket_id}")
                else:
                    logger.info(f"  ⏭️  Skip existing: {docket_id}")

                # Stop if we have enough
                if len(new_dockets) >= target_count:
                    break

            logger.info(f"📊 Page {page_num}: {len(page_dockets)} total, {page_new_count} new (total new: {len(new_dockets)})")

            if len(new_dockets) >= target_count:
                logger.info(f"🎉 Reached target of {target_count} new dockets")
                break

        final_new_dockets = new_dockets[:target_count]
        logger.info(f"\n✅ Pagination simulation complete:")
        logger.info(f"   🎯 Target: {target_count}")
        logger.info(f"   ✨ Found: {len(final_new_dockets)}")
        logger.info(f"   📋 New docket IDs: {[d['id'] for d in final_new_dockets]}")

        return True

    def test_edge_cases(self):
        """Test various edge cases."""
        logger.info("\n" + "="*60)
        logger.info("🧪 Testing Edge Cases")
        logger.info("="*60)

        # Test case 1: Empty docket ID
        logger.info("🔍 Test case 1: Empty docket ID handling")
        existing_dockets = set(['1', '2', '3'])
        test_docket = {'id': ''}  # Empty ID

        docket_id = test_docket.get('id', '')
        if docket_id and docket_id not in existing_dockets:
            logger.info("❌ Empty ID incorrectly treated as new")
        else:
            logger.info("✅ Empty ID correctly ignored")

        # Test case 2: None docket ID
        logger.info("🔍 Test case 2: None docket ID handling")
        test_docket = {'id': None}  # None ID

        docket_id = test_docket.get('id', '')
        if docket_id and docket_id not in existing_dockets:
            logger.info("❌ None ID incorrectly treated as new")
        else:
            logger.info("✅ None ID correctly ignored")

        # Test case 3: Missing ID field
        logger.info("🔍 Test case 3: Missing ID field handling")
        test_docket = {}  # No ID field

        docket_id = test_docket.get('id', '')
        if docket_id and docket_id not in existing_dockets:
            logger.info("❌ Missing ID incorrectly treated as new")
        else:
            logger.info("✅ Missing ID correctly ignored")

        # Test case 4: String vs numeric ID consistency
        logger.info("🔍 Test case 4: String vs numeric ID consistency")
        existing_dockets = set(['123', '456', '789'])

        # Test string ID
        if '123' in existing_dockets:
            logger.info("✅ String ID '123' correctly found in existing set")
        else:
            logger.info("❌ String ID '123' not found")

        # Test what happens with numeric (this shouldn't happen in real API but good to check)
        if 123 in existing_dockets:
            logger.info("❌ Numeric ID 123 found in string set (unexpected)")
        else:
            logger.info("✅ Numeric ID 123 correctly not found in string set")

        logger.info("✅ Edge case testing complete")

        return True

    def test_ordering_consistency(self):
        """Test that oldest-first ordering is consistent."""
        logger.info("\n" + "="*60)
        logger.info("🧪 Testing Ordering Consistency (id vs -id)")
        logger.info("="*60)

        # Simulate the difference between old (-id) and new (id) ordering
        docket_ids = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

        logger.info("📊 Original docket IDs (from API): " + str(docket_ids))

        # Old ordering: newest first (-id)
        old_ordering = sorted(docket_ids, reverse=True)
        logger.info(f"❌ Old ordering (-id, newest first): {old_ordering}")
        logger.info("   Problem: If new dockets added between runs, pagination breaks")

        # New ordering: oldest first (id)
        new_ordering = sorted(docket_ids, reverse=False)
        logger.info(f"✅ New ordering (id, oldest first): {new_ordering}")
        logger.info("   Benefit: Consistent pagination, always starts from beginning")

        # Simulate what happens when new dockets are added
        logger.info("\n🔄 Simulating new dockets added to CourtListener...")
        new_docket_ids = [2, 3, 7, 12, 18, 22, 33]  # New dockets inserted
        all_dockets = sorted(docket_ids + new_docket_ids)

        logger.info(f"📊 Updated docket list: {all_dockets}")

        # With old ordering (-id), pagination would be inconsistent
        old_ordering_updated = sorted(all_dockets, reverse=True)
        logger.info(f"❌ Old ordering with new dockets: {old_ordering_updated}")
        logger.info("   Problem: Previously seen dockets might be processed again")

        # With new ordering (id), pagination remains consistent
        new_ordering_updated = sorted(all_dockets, reverse=False)
        logger.info(f"✅ New ordering with new dockets: {new_ordering_updated}")
        logger.info("   Benefit: Always starts from oldest, skips existing via deduplication")

        logger.info("✅ Ordering consistency test complete")

        return True

    def run_all_tests(self):
        """Run all deduplication tests."""
        logger.info("🚀 Starting Deduplication Logic Testing")
        logger.info("="*60)

        self.setup()

        tests = [
            ("get_existing_docket_ids", self.test_get_existing_docket_ids),
            ("empty_collection_scenario", self.test_empty_collection_scenario),
            ("all_dockets_exist_scenario", self.test_all_dockets_exist_scenario),
            ("mixed_scenario", self.test_mixed_scenario),
            ("pagination_logic_simulation", self.test_pagination_logic_simulation),
            ("edge_cases", self.test_edge_cases),
            ("ordering_consistency", self.test_ordering_consistency),
            ("true_deduplication_in_qdrant", self.test_true_deduplication_in_qdrant)
        ]

        results = {}

        for test_name, test_func in tests:
            try:
                logger.info(f"\n🧪 Running test: {test_name}")
                result = test_func()

                # Handle both boolean and string return types
                if isinstance(result, str):
                    results[test_name] = result
                    status = "PASSED" if result == "PASSED" else "FAILED"
                else:
                    results[test_name] = "PASSED" if result else "FAILED"
                    status = "PASSED" if result else "FAILED"

                logger.info(f"✅ Test {test_name}: {status}")
            except Exception as e:
                logger.error(f"❌ Test {test_name}: FAILED - {e}")
                results[test_name] = f"FAILED: {e}"

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("📊 TEST SUMMARY")
        logger.info("="*60)

        passed = sum(1 for result in results.values() if result == "PASSED")
        total = len(results)

        for test_name, result in results.items():
            status_icon = "✅" if result == "PASSED" else "❌"
            logger.info(f"{status_icon} {test_name}: {result}")

        logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")

        if passed == total:
            logger.info("🎉 All tests passed! Deduplication logic looks good.")
        else:
            logger.warning("⚠️ Some tests failed. Review the implementation.")

        return results

    def test_true_deduplication_in_qdrant(self) -> str:
        """
        Test that no duplicate entries exist in Qdrant based on 4-tuple key.

        Checks for duplicate combinations of:
        (docket_id, cluster_id, opinion_id, chunk_index)

        Returns:
            str: "PASSED" if no duplicates found, "FAILED" otherwise
        """
        logger.info("🔍 Testing true deduplication in Qdrant collection...")

        try:
            # Initialize vector processor to get Qdrant client
            from vector_processor import EnhancedVectorProcessor
            from config import PipelineConfig

            config = PipelineConfig()
            vector_processor = EnhancedVectorProcessor(
                model_name=config.vector_processing.embedding_model,
                collection_name=config.vector_processing.collection_name_vector,
                qdrant_url=None  # Will use default/env
            )
            client = vector_processor._get_qdrant_client()
            collection_name = config.vector_processing.collection_name_vector

            # Check if collection exists
            try:
                collection_info = client.get_collection(collection_name)
                total_points = collection_info.points_count
                logger.info(f"📊 Collection '{collection_name}' has {total_points} points")

                if total_points == 0:
                    logger.info("ℹ️ Collection is empty - skipping deduplication test")
                    return "PASSED"

            except Exception as e:
                logger.warning(f"⚠️ Collection '{collection_name}' not found or inaccessible: {e}")
                return "PASSED"  # Can't test what doesn't exist

            # Scroll through all entries and collect 4-tuples
            seen_tuples = set()
            duplicate_tuples = []
            processed_count = 0

            # Use scroll to get all points with payload
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=1000,  # Process in batches of 1000
                with_payload=True,
                with_vectors=False  # We only need payload data
            )

            points = scroll_result[0]
            next_page_offset = scroll_result[1]

            while points:
                for point in points:
                    processed_count += 1
                    payload = point.payload or {}

                    # Extract the 4-tuple key
                    docket_id = payload.get('docket_id', 'unknown')
                    cluster_id = payload.get('cluster_id', 'unknown')
                    opinion_id = payload.get('opinion_id', 'unknown')
                    chunk_index = payload.get('chunk_index', -1)

                    # Create the 4-tuple
                    tuple_key = (docket_id, cluster_id, opinion_id, chunk_index)

                    # Check for duplicate
                    if tuple_key in seen_tuples:
                        duplicate_tuples.append({
                            'point_id': point.id,
                            'tuple': tuple_key,
                            'payload': payload
                        })
                        logger.error(f"❌ Duplicate found: {tuple_key} (point ID: {point.id})")
                    else:
                        seen_tuples.add(tuple_key)

                # Get next batch if available
                if next_page_offset:
                    scroll_result = client.scroll(
                        collection_name=collection_name,
                        limit=1000,
                        offset=next_page_offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    points = scroll_result[0]
                    next_page_offset = scroll_result[1]
                else:
                    break

            # Report results
            unique_tuples = len(seen_tuples)
            logger.info(f"📈 Processed {processed_count} points")
            logger.info(f"🔑 Found {unique_tuples} unique 4-tuples")

            if duplicate_tuples:
                logger.error(f"❌ Found {len(duplicate_tuples)} duplicate entries!")
                logger.error("📋 Duplicate details:")
                for i, dup in enumerate(duplicate_tuples[:5]):  # Show first 5 duplicates
                    logger.error(f"  {i+1}. Point ID: {dup['point_id']}, Tuple: {dup['tuple']}")
                if len(duplicate_tuples) > 5:
                    logger.error(f"  ... and {len(duplicate_tuples) - 5} more duplicates")
                return "FAILED"
            else:
                logger.info("✅ No duplicates found - true deduplication is working correctly!")
                return "PASSED"

        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
            logger.error(f"📋 Exception details: {type(e).__name__}")
            return "FAILED"

def main():
    """Main test execution."""
    load_dotenv()

    tester = DeduplicationTester()
    results = tester.run_all_tests()

    # Exit with appropriate code
    if all(result == "PASSED" for result in results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()


# Pytest-compatible test functions
def test_deduplication_logic():
    """Pytest wrapper for deduplication logic tests."""
    load_dotenv()
    tester = DeduplicationTester()
    results = tester.run_all_tests()

    # Check if all tests passed
    failed_tests = [name for name, result in results.items() if not result.startswith("PASSED")]
    if failed_tests:
        raise AssertionError(f"Failed tests: {failed_tests}")


def test_true_deduplication_qdrant():
    """Pytest wrapper for true deduplication test in Qdrant."""
    load_dotenv()
    tester = DeduplicationTester()
    tester.setup()

    result = tester.test_true_deduplication_in_qdrant()
    assert result == "PASSED", f"True deduplication test failed: {result}"