#!/usr/bin/env python3
"""
Direct Test of Updated Pagination Logic

This script directly tests the _fetch_all_dockets_paginated method
to verify the deduplication and ordering changes work correctly.
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
from config import PipelineConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def mock_fetch_with_retry(url, headers=None, timeout=30, max_retries=3, delay=5):
    """Mock function to simulate API responses."""
    logger.info(f"🌐 Mock API call to: {url}")

    # Extract cursor from URL
    cursor = None
    if 'cursor=page2' in url:
        cursor = 'page2'
    elif 'cursor=page3' in url:
        cursor = 'page3'

    # Return different pages based on cursor
    if cursor is None:  # First page
        return {
            'results': [{'id': str(i)} for i in range(1, 21)],  # IDs 1-20
            'next': 'https://api.example.com/dockets/?cursor=page2'
        }
    elif cursor == 'page2':  # Second page
        return {
            'results': [{'id': str(i)} for i in range(21, 41)],  # IDs 21-40
            'next': 'https://api.example.com/dockets/?cursor=page3'
        }
    elif cursor == 'page3':  # Third page
        return {
            'results': [{'id': str(i)} for i in range(41, 61)],  # IDs 41-60
            'next': None  # Last page
        }
    else:
        return {'results': [], 'next': None}

def test_pagination_scenarios():
    """Test various pagination scenarios."""
    logger.info("🧪 Testing Direct Pagination Logic")
    logger.info("="*60)

    # Initialize pipeline
    config = PipelineConfig()
    pipeline = LegalDocumentPipeline(config)

    test_scenarios = [
        {
            'name': 'Empty collection (first run)',
            'existing_dockets': set(),
            'expected_first_5': ['1', '2', '3', '4', '5']
        },
        {
            'name': 'Some existing dockets',
            'existing_dockets': set(['5', '10', '15', '25', '30']),
            'expected_first_5': ['1', '2', '3', '4', '6']  # Skip 5
        },
        {
            'name': 'Many existing dockets',
            'existing_dockets': set([str(i) for i in range(1, 31)]),  # 1-30 exist
            'expected_first_5': ['31', '32', '33', '34', '35']  # Start from 31
        }
    ]

    for scenario in test_scenarios:
        logger.info(f"\n🔍 Testing scenario: {scenario['name']}")
        logger.info(f"📊 Existing dockets: {len(scenario['existing_dockets'])} total")

        with patch('main.fetch_with_retry', side_effect=mock_fetch_with_retry):
            try:
                # Test with a request for 5 dockets
                new_dockets = pipeline._fetch_all_dockets_paginated(
                    court='scotus',
                    num_dockets=5,
                    existing_dockets=scenario['existing_dockets']
                )

                actual_ids = [d['id'] for d in new_dockets]
                expected_ids = scenario['expected_first_5']

                logger.info(f"📋 Expected: {expected_ids}")
                logger.info(f"📋 Actual:   {actual_ids}")

                if actual_ids == expected_ids:
                    logger.info("✅ Scenario PASSED")
                else:
                    logger.error("❌ Scenario FAILED")
                    return False

            except Exception as e:
                logger.error(f"❌ Scenario failed with error: {e}")
                return False

    logger.info("\n✅ All pagination scenarios passed!")
    return True

def test_cursor_handling():
    """Test cursor extraction and handling."""
    logger.info("\n🧪 Testing Cursor Handling")
    logger.info("="*40)

    config = PipelineConfig()
    pipeline = LegalDocumentPipeline(config)

    # Mock different cursor responses
    def mock_cursor_responses(url, headers=None, timeout=30, max_retries=3, delay=5):
        if 'cursor=' not in url:  # First call
            return {
                'results': [{'id': '1'}, {'id': '2'}],
                'next': 'https://api.example.com/dockets/?cursor=abc123'
            }
        elif 'cursor=abc123' in url:  # Second call
            return {
                'results': [{'id': '3'}, {'id': '4'}],
                'next': 'https://api.example.com/dockets/?cursor=def456'
            }
        elif 'cursor=def456' in url:  # Third call
            return {
                'results': [{'id': '5'}, {'id': '6'}],
                'next': None  # End of data
            }
        else:
            return {'results': [], 'next': None}

    with patch('main.fetch_with_retry', side_effect=mock_cursor_responses):
        try:
            new_dockets = pipeline._fetch_all_dockets_paginated(
                court='scotus',
                num_dockets=5,
                existing_dockets=set()
            )

            actual_ids = [d['id'] for d in new_dockets]
            expected_ids = ['1', '2', '3', '4', '5']

            if actual_ids == expected_ids:
                logger.info("✅ Cursor handling test PASSED")
                return True
            else:
                logger.error(f"❌ Expected {expected_ids}, got {actual_ids}")
                return False

        except Exception as e:
            logger.error(f"❌ Cursor handling test failed: {e}")
            return False

def test_ordering_verification():
    """Verify that oldest-first ordering is maintained."""
    logger.info("\n🧪 Testing Ordering Verification")
    logger.info("="*40)

    # Create mock data with non-sequential IDs to test ordering
    def mock_ordering_responses(url, headers=None, timeout=30, max_retries=3, delay=5):
        if 'cursor=' not in url:  # First call
            # Return IDs in order, but not sequential (simulating real API)
            return {
                'results': [
                    {'id': '100'}, {'id': '105'}, {'id': '110'},
                    {'id': '115'}, {'id': '120'}
                ],
                'next': 'https://api.example.com/dockets/?cursor=next'
            }
        else:  # Second call
            return {
                'results': [
                    {'id': '125'}, {'id': '130'}, {'id': '135'},
                    {'id': '140'}, {'id': '145'}
                ],
                'next': None
            }

    config = PipelineConfig()
    pipeline = LegalDocumentPipeline(config)

    with patch('main.fetch_with_retry', side_effect=mock_ordering_responses):
        try:
            # Test with some existing dockets
            existing_dockets = set(['105', '120', '130'])

            new_dockets = pipeline._fetch_all_dockets_paginated(
                court='scotus',
                num_dockets=5,
                existing_dockets=existing_dockets
            )

            actual_ids = [d['id'] for d in new_dockets]
            # Should get: 100, 110, 115, 125, 135 (skipping 105, 120, 130)
            expected_ids = ['100', '110', '115', '125', '135']

            if actual_ids == expected_ids:
                logger.info("✅ Ordering verification test PASSED")
                logger.info(f"📋 Correctly skipped existing: {existing_dockets}")
                logger.info(f"📋 Returned in order: {actual_ids}")
                return True
            else:
                logger.error(f"❌ Expected {expected_ids}, got {actual_ids}")
                return False

        except Exception as e:
            logger.error(f"❌ Ordering verification test failed: {e}")
            return False

def main():
    """Run all direct pagination tests."""
    logger.info("🚀 Starting Direct Pagination Tests")
    logger.info("="*60)

    load_dotenv()

    tests = [
        ("pagination_scenarios", test_pagination_scenarios),
        ("cursor_handling", test_cursor_handling),
        ("ordering_verification", test_ordering_verification)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASSED" if result else "FAILED"
        except Exception as e:
            logger.error(f"❌ Test {test_name} failed with exception: {e}")
            results[test_name] = f"FAILED: {e}"

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 DIRECT PAGINATION TEST SUMMARY")
    logger.info("="*60)

    passed = sum(1 for result in results.values() if result == "PASSED")
    total = len(results)

    for test_name, result in results.items():
        status_icon = "✅" if result == "PASSED" else "❌"
        logger.info(f"{status_icon} {test_name}: {result}")

    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All direct pagination tests passed!")
        logger.info("✅ Updated deduplication logic is working correctly!")
    else:
        logger.warning("⚠️ Some tests failed. Check the implementation.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)