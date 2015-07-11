#!/usr/bin/python
"""Combines both the blur and blend scripts into one."""
import blur
import blend


def main():
    """Do both operations on all imags."""
    print 'Blurring'
    print '-' * 20
    blur.main()

    print ''
    print 'Blending'
    print '-' * 20
    blend.main()

if __name__ == "__main__":
    main()
