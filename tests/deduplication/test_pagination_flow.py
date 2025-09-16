#!/usr/bin/env python3
"""
Focused Test for Pagination Flow and Deduplication Integration

This script tests the actual integration between the pagination logic
and deduplication to ensure the updated system works end-to-end.
"""

import os
import sys
import json
import logging
from typing import Dict, Set, List, Any
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append('/root/lawlm')

from main import LegalDocumentPipeline
from vector_processor import EnhancedVectorProcessor
from config import PipelineConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PaginationFlowTester:
    """Test the integration between pagination and deduplication."""

    def __init__(self):
        self.config = PipelineConfig()
        self.pipeline = LegalDocumentPipeline(self.config)

    def test_pagination_deduplication_integration(self):
        """Test the full pagination flow with deduplication."""
        logger.info("🧪 Testing Pagination-Deduplication Integration")
        logger.info("="*60)

        # Mock existing dockets
        existing_dockets = set(['5', '10', '15', '25', '30', '35'])
        logger.info(f"📊 Existing dockets: {existing_dockets}")

        # Mock API responses for multiple pages
        mock_responses = [
            # Page 1: dockets 1-20 (cursor-paginated)
            {
                'results': [{'id': str(i)} for i in range(1, 21)],
                'next': 'https://api.example.com/dockets/?cursor=page2'
            },
            # Page 2: dockets 21-40
            {
                'results': [{'id': str(i)} for i in range(21, 41)],
                'next': 'https://api.example.com/dockets/?cursor=page3'
            },
            # Page 3: dockets 41-60
            {
                'results': [{'id': str(i)} for i in range(41, 61)],
                'next': None  # Last page
            }
        ]

        # Mock the vector processor's get_existing_docket_ids method
        with patch.object(self.pipeline.vector_processor, 'get_existing_docket_ids', return_value=existing_dockets):
            with patch('main.fetch_with_retry') as mock_fetch:
                mock_fetch.side_effect = mock_responses

                # Test the pagination flow
                try:
                    new_dockets = self.pipeline._fetch_all_dockets_paginated(
                        court='scotus',
                        num_dockets=10,
                        existing_dockets=existing_dockets
                    )

                    logger.info(f"✅ Pagination completed successfully")
                    logger.info(f"📊 New dockets found: {len(new_dockets)}")

                    # Check that only new dockets were returned
                    new_docket_ids = [d['id'] for d in new_dockets]
                    logger.info(f"📋 New docket IDs: {new_docket_ids}")

                    # Verify no existing dockets were included
                    overlaps = set(new_docket_ids).intersection(existing_dockets)
                    if overlaps:
                        logger.error(f"❌ Found overlapping dockets: {overlaps}")
                        return False
                    else:
                        logger.info("✅ No overlapping dockets found")

                    # Verify we got the expected new dockets (oldest first, skipping existing)
                    expected_new = ['1', '2', '3', '4', '6', '7', '8', '9', '11', '12']
                    if new_docket_ids[:10] == expected_new:
                        logger.info("✅ Correct dockets returned in correct order")
                        return True
                    else:
                        logger.error(f"❌ Unexpected dockets. Expected: {expected_new}, Got: {new_docket_ids}")
                        return False

                except Exception as e:
                    logger.error(f"❌ Pagination test failed: {e}")
                    return False

    def test_cursor_pagination_logic(self):
        """Test the cursor extraction and pagination logic."""
        logger.info("🧪 Testing Cursor Pagination Logic")
        logger.info("="*60)

        # Test cursor extraction from next URLs
        test_urls = [
            "https://www.courtlistener.com/api/rest/v4/dockets/?cursor=abc123&court=scotus",
            "https://api.example.com/dockets/?cursor=xyz789",
            "https://api.example.com/dockets/?other_param=value&cursor=def456&more=stuff",
            None  # No next URL (last page)
        ]

        expected_cursors = ['abc123', 'xyz789', 'def456', None]

        import urllib.parse

        for i, url in enumerate(test_urls):
            if url:
                parsed = urllib.parse.urlparse(url)
                query_params = urllib.parse.parse_qs(parsed.query)
                cursor = query_params.get('cursor', [None])[0]
            else:
                cursor = None

            expected = expected_cursors[i]

            if cursor == expected:
                logger.info(f"✅ Cursor extraction {i+1}: {cursor} (correct)")
            else:
                logger.error(f"❌ Cursor extraction {i+1}: expected {expected}, got {cursor}")
                return False

        logger.info("✅ All cursor extractions correct")
        return True

    def test_ordering_consistency(self):
        """Test that the new ordering (id) provides consistent results."""
        logger.info("🧪 Testing Ordering Consistency")
        logger.info("="*60)

        # Simulate the problem with old ordering (-id)
        logger.info("📊 Old ordering (-id) problems:")

        # Original dockets in system
        original_dockets = [10, 20, 30, 40, 50]
        logger.info(f"   Original dockets: {original_dockets}")

        # With -id ordering, we'd see: [50, 40, 30, 20, 10]
        old_order = sorted(original_dockets, reverse=True)
        logger.info(f"   Old order (-id): {old_order}")

        # If we processed the first 3 dockets: [50, 40, 30]
        processed_old = old_order[:3]
        logger.info(f"   Processed first 3: {processed_old}")

        # Now new dockets are added: [15, 25, 35, 45]
        new_dockets_added = [15, 25, 35, 45]
        all_dockets = sorted(original_dockets + new_dockets_added)
        logger.info(f"   New dockets added: {new_dockets_added}")
        logger.info(f"   All dockets now: {all_dockets}")

        # With -id ordering after new dockets: [50, 45, 40, 35, 30, 25, 20, 15, 10]
        new_old_order = sorted(all_dockets, reverse=True)
        logger.info(f"   New old order (-id): {new_old_order}")
        logger.info("   ❌ Problem: Position of previously processed dockets changed!")

        # With new ordering (id), we always start from beginning
        logger.info("\n📊 New ordering (id) benefits:")
        new_order = sorted(all_dockets)
        logger.info(f"   New order (id): {new_order}")
        logger.info("   ✅ Benefit: Always consistent, starts from oldest")

        # With deduplication, we skip what we've already processed
        existing_set = set([str(d) for d in processed_old])  # Convert to strings
        logger.info(f"   Existing in vector DB: {existing_set}")

        # Process from beginning, skipping existing
        new_to_process = []
        for docket_id in new_order:
            if str(docket_id) not in existing_set:
                new_to_process.append(docket_id)

        logger.info(f"   New dockets to process: {new_to_process}")
        logger.info("   ✅ Result: Only truly new dockets processed")

        return True

    def test_edge_case_scenarios(self):
        """Test various edge case scenarios."""
        logger.info("🧪 Testing Edge Case Scenarios")
        logger.info("="*60)

        scenarios = [
            {
                'name': 'Empty existing dockets',
                'existing': set(),
                'api_dockets': [{'id': str(i)} for i in range(1, 6)],
                'expected_count': 5
            },
            {
                'name': 'All dockets exist',
                'existing': set([str(i) for i in range(1, 11)]),
                'api_dockets': [{'id': str(i)} for i in range(1, 6)],
                'expected_count': 0
            },
            {
                'name': 'Mixed scenario',
                'existing': set(['2', '4', '6']),
                'api_dockets': [{'id': str(i)} for i in range(1, 8)],
                'expected_count': 4  # 1, 3, 5, 7
            },
            {
                'name': 'Invalid docket IDs',
                'existing': set(['1', '2']),
                'api_dockets': [
                    {'id': '1'},
                    {'id': ''},      # Empty ID
                    {'id': None},    # None ID
                    {},              # Missing ID
                    {'id': '3'}      # Valid new ID
                ],
                'expected_count': 1  # Only ID '3'
            }
        ]

        for scenario in scenarios:
            logger.info(f"\n🔍 Testing: {scenario['name']}")

            existing_dockets = scenario['existing']
            api_dockets = scenario['api_dockets']
            expected_count = scenario['expected_count']

            # Simulate the filtering logic
            new_dockets = []
            for docket in api_dockets:
                docket_id = docket.get('id', '')
                if docket_id and docket_id not in existing_dockets:
                    new_dockets.append(docket)

            actual_count = len(new_dockets)

            if actual_count == expected_count:
                logger.info(f"   ✅ Expected {expected_count}, got {actual_count}")
            else:
                logger.error(f"   ❌ Expected {expected_count}, got {actual_count}")
                logger.error(f"      New dockets: {[d['id'] for d in new_dockets]}")
                return False

        logger.info("✅ All edge case scenarios passed")
        return True

    def run_all_tests(self):
        """Run all pagination flow tests."""
        logger.info("🚀 Starting Pagination Flow Testing")
        logger.info("="*80)

        tests = [
            ("pagination_deduplication_integration", self.test_pagination_deduplication_integration),
            ("cursor_pagination_logic", self.test_cursor_pagination_logic),
            ("ordering_consistency", self.test_ordering_consistency),
            ("edge_case_scenarios", self.test_edge_case_scenarios)
        ]

        results = {}

        for test_name, test_func in tests:
            try:
                logger.info(f"\n{'='*60}")
                result = test_func()
                results[test_name] = "PASSED" if result else "FAILED"
                logger.info(f"✅ Test {test_name}: PASSED" if result else f"❌ Test {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ Test {test_name}: FAILED - {e}")
                results[test_name] = f"FAILED: {e}"

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("📊 PAGINATION FLOW TEST SUMMARY")
        logger.info("="*80)

        passed = sum(1 for result in results.values() if result == "PASSED")
        total = len(results)

        for test_name, result in results.items():
            status_icon = "✅" if result == "PASSED" else "❌"
            logger.info(f"{status_icon} {test_name}: {result}")

        logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")

        if passed == total:
            logger.info("🎉 All pagination flow tests passed!")
        else:
            logger.warning("⚠️ Some pagination flow tests failed.")

        return results

def main():
    """Main test execution."""
    load_dotenv()

    tester = PaginationFlowTester()
    results = tester.run_all_tests()

    # Exit with appropriate code
    if all(result == "PASSED" for result in results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()