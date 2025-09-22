import cv2
import numpy as np
import matplotlib.pyplot as plt

# find the matching sift points using open cvs sift matcher
def findsiftpoints(impath1, impath2, num_best_matches=20):
    img1_gray = cv2.imread(impath1, cv2.IMREAD_GRAYSCALE)
    img2_gray = cv2.imread(impath2, cv2.IMREAD_GRAYSCALE)
    img1_color = cv2.imread(impath1)
    img2_color = cv2.imread(impath2)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m,n in matches if m.distance < 0.7 * n.distance]
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    best_matches = good_matches[:num_best_matches]

    return good_matches, best_matches, kp1, kp2, img1_color, img2_color

# finding A matrix for the homogeneous solution
def find_A_matrix(matches, kp1, kp2):
    A_rows = []
    for m in matches:
        x, y = kp1[m.queryIdx].pt
        xp, yp = kp2[m.trainIdx].pt
        A_rows.append([-x, -y, -1, 0, 0, 0, xp*x, xp*y, xp])
        A_rows.append([0, 0, 0, -x, -y, -1, yp*x, yp*y, yp])
    return np.array(A_rows, dtype=float)

# Finding the null space solution for A
def solve_homography_from_A(A):
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1,:]
    H = h.reshape(3,3)
    H = H / H[2, 2]

    # returns the homogeneous matrix and singular values of A
    return H, S

# finding the points which agree with the homogeneous matrix calculated based on a threshold
def find_inliers(H, matches, kp1, kp2, threshold=5):
    inliers = []
    for m in matches:
        x, y = kp1[m.queryIdx].pt
        xp, yp = kp2[m.trainIdx].pt
        pt = np.array([x, y, 1.0])
        mapped = H @ pt
        mapped /= mapped[2]
        error = np.linalg.norm(np.array([xp, yp]) - mapped[:2])
        if error <= threshold:
            inliers.append(m)
    return inliers

# create a mosaic of both the images using the calculated homogeneous matrix
def create_mosaic(H, img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img1, H)
    corners_img2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    all_corners = np.vstack((warped_corners, corners_img2))
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation = [-xmin, -ymin]

    H_translation = np.array([[1,0,translation[0]],
                              [0,1,translation[1]],
                              [0,0,1]])
    
    mosaic = cv2.warpPerspective(img1, H_translation @ H, (xmax-xmin, ymax-ymin))
    mosaic[translation[1]:translation[1]+h2, translation[0]:translation[0]+w2] = img2
    return mosaic

def main(image1_path, image2_path, num_best_matches=8):
    good_matches, best_matches, kp1, kp2, img1_color, img2_color = findsiftpoints(
        image1_path, image2_path, num_best_matches
    )

    print(f"Total good matches: {len(good_matches)}")
    print(f"Top {len(best_matches)} matches used for homography.")

    A = find_A_matrix(best_matches, kp1, kp2)
    print("\nMatrix A shape:", A.shape)

    H, S = solve_homography_from_A(A)
    print("\nSingular values of A:\n", S)
    print("\nHomography H from null-space of A:\n", H)

    inliers = find_inliers(H, good_matches, kp1, kp2, threshold=100)
    print(f"\nNumber of inliers: {len(inliers)} / {len(good_matches)}")

    img_matches_all = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good_matches, None,
                                      matchColor=(200,200,200), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_inliers = cv2.drawMatches(img1_color, kp1, img2_color, kp2, inliers, None,
                                         matchColor=(0,255,0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.title("Best matches (gray)")
    plt.imshow(cv2.cvtColor(img_matches_all, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title(f"Inliers (green), count={len(inliers)}")
    plt.imshow(cv2.cvtColor(img_matches_inliers, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    cv2.imwrite("matches_all.png", img_matches_all)
    cv2.imwrite("matches_inliers.png", img_matches_inliers)

    mosaic = create_mosaic(H, img1_color, img2_color)
    plt.figure(figsize=(15,10))
    plt.title("Mosaic")
    plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example usage:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SIFT-based image mosaic")
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    parser.add_argument("--matches", type=int, default=8, help="Number of best matches to use")
    args = parser.parse_args()

    main(args.image1, args.image2, num_best_matches=args.matches)
